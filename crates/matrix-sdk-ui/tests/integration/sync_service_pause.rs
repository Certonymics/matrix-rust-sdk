// Copyright 2026 The Matrix.org Foundation C.I.C.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! `SyncService::stop` then `Client::pause`, the order `Client::pause`
//! documents, while a sliding sync response is still being handled.

use std::{
    sync::{Arc, Mutex},
    time::Duration,
};

use async_trait::async_trait;
use matrix_sdk::{SqliteEventCacheStore, config::StoreConfig, test_utils::mocks::MatrixMockServer};
use matrix_sdk_base::event_cache::{
    Event, Gap,
    store::{DynEventCacheStore, EventCacheStore, EventCacheStoreError, IntoEventCacheStore},
};
use matrix_sdk_common::{
    cross_process_lock::{
        CrossProcessLockConfig, CrossProcessLockGeneration, MappedCrossProcessLockState,
    },
    linked_chunk::{
        ChunkContent, ChunkIdentifier, ChunkIdentifierGenerator, ChunkMetadata, LinkedChunkId,
        Position, RawChunk, Update,
    },
};
use matrix_sdk_test::async_test;
use matrix_sdk_ui::sync_service::SyncService;
use ruma::{EventId, OwnedEventId, RoomId, event_id, events::relation::RelationType, room_id};
use serde_json::{Value, json};
use tokio::{spawn, sync::Notify, time::timeout};
use wiremock::{Mock, Request, ResponseTemplate};

use crate::sliding_sync::{PartialSlidingSyncRequest, SlidingSyncMatcher};

/// Holds the event cache's store write of one event until released, so the
/// test decides what happens while that write is in flight.
#[derive(Default)]
struct HeldWrite {
    event_id: Mutex<Option<OwnedEventId>>,
    held: Notify,
    released: Notify,
    /// The held write has returned, whether it succeeded or failed.
    done: Notify,
}

impl HeldWrite {
    /// Returns whether the write was held.
    async fn wait_if_armed(&self, updates: &[Update<Event, Gap>]) -> bool {
        let carries = |id: &EventId| {
            updates.iter().any(|update| match update {
                Update::PushItems { items, .. } => {
                    items.iter().any(|event| event.event_id().as_deref() == Some(id))
                }
                _ => false,
            })
        };
        let hold = {
            let mut armed = self.event_id.lock().unwrap();
            let hold = armed.as_deref().is_some_and(carries);
            if hold {
                armed.take();
            }
            hold
        };
        if hold {
            self.held.notify_one();
            self.released.notified().await;
        }
        hold
    }
}

/// A SQLite event cache store whose writes [`HeldWrite`] can hold.
#[derive(Debug)]
struct HoldingStore {
    inner: Arc<DynEventCacheStore>,
    held: Arc<HeldWrite>,
}

impl std::fmt::Debug for HeldWrite {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HeldWrite").finish_non_exhaustive()
    }
}

#[cfg_attr(target_family = "wasm", async_trait(?Send))]
#[cfg_attr(not(target_family = "wasm"), async_trait)]
impl EventCacheStore for HoldingStore {
    type Error = EventCacheStoreError;

    async fn try_take_leased_lock(
        &self,
        lease_duration_ms: u32,
        key: &str,
        holder: &str,
    ) -> Result<Option<CrossProcessLockGeneration>, Self::Error> {
        self.inner.try_take_leased_lock(lease_duration_ms, key, holder).await
    }

    async fn handle_linked_chunk_updates(
        &self,
        linked_chunk_id: LinkedChunkId<'_>,
        updates: Vec<Update<Event, Gap>>,
    ) -> Result<(), Self::Error> {
        let held = self.held.wait_if_armed(&updates).await;
        let written = self.inner.handle_linked_chunk_updates(linked_chunk_id, updates).await;
        if held {
            self.held.done.notify_one();
        }
        written
    }

    async fn load_all_chunks(
        &self,
        linked_chunk_id: LinkedChunkId<'_>,
    ) -> Result<Vec<RawChunk<Event, Gap>>, Self::Error> {
        self.inner.load_all_chunks(linked_chunk_id).await
    }

    async fn load_all_chunks_metadata(
        &self,
        linked_chunk_id: LinkedChunkId<'_>,
    ) -> Result<Vec<ChunkMetadata>, Self::Error> {
        self.inner.load_all_chunks_metadata(linked_chunk_id).await
    }

    async fn load_last_chunk(
        &self,
        linked_chunk_id: LinkedChunkId<'_>,
    ) -> Result<(Option<RawChunk<Event, Gap>>, ChunkIdentifierGenerator), Self::Error> {
        self.inner.load_last_chunk(linked_chunk_id).await
    }

    async fn load_previous_chunk(
        &self,
        linked_chunk_id: LinkedChunkId<'_>,
        before_chunk_identifier: ChunkIdentifier,
    ) -> Result<Option<RawChunk<Event, Gap>>, Self::Error> {
        self.inner.load_previous_chunk(linked_chunk_id, before_chunk_identifier).await
    }

    async fn clear_all_linked_chunks(&self) -> Result<(), Self::Error> {
        self.inner.clear_all_linked_chunks().await
    }

    async fn filter_duplicated_events(
        &self,
        linked_chunk_id: LinkedChunkId<'_>,
        events: Vec<OwnedEventId>,
    ) -> Result<Vec<(OwnedEventId, Position)>, Self::Error> {
        self.inner.filter_duplicated_events(linked_chunk_id, events).await
    }

    async fn find_event(
        &self,
        room_id: &RoomId,
        event_id: &EventId,
    ) -> Result<Option<Event>, Self::Error> {
        self.inner.find_event(room_id, event_id).await
    }

    async fn find_event_relations(
        &self,
        room_id: &RoomId,
        event_id: &EventId,
        filter: Option<&[RelationType]>,
    ) -> Result<Vec<(Event, Option<Position>)>, Self::Error> {
        self.inner.find_event_relations(room_id, event_id, filter).await
    }

    async fn get_room_events(
        &self,
        room_id: &RoomId,
        event_type: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<Vec<Event>, Self::Error> {
        self.inner.get_room_events(room_id, event_type, session_id).await
    }

    async fn save_event(&self, room_id: &RoomId, event: Event) -> Result<(), Self::Error> {
        self.inner.save_event(room_id, event).await
    }

    async fn close(&self) -> Result<(), Self::Error> {
        self.inner.close().await
    }

    async fn reopen(&self) -> Result<(), Self::Error> {
        self.inner.reopen().await
    }

    async fn optimize(&self) -> Result<(), Self::Error> {
        self.inner.optimize().await
    }

    async fn get_size(&self) -> Result<Option<usize>, Self::Error> {
        self.inner.get_size().await
    }
}

fn message(event_id: &EventId) -> Value {
    json!({
        "type": "m.room.message",
        "event_id": event_id,
        "sender": "@bob:localhost",
        "origin_server_ts": 1,
        "content": { "msgtype": "m.text", "body": event_id },
    })
}

/// `SyncService::stop` returned while the event cache was still writing a
/// received response, so the `Client::pause` that followed closed the event
/// cache store under that write. When it was the room's first write, every
/// later write for the room failed on its foreign key, and the room's events
/// stopped reaching the store until the process restarted.
///
/// The room's first write is held while `stop` and `pause` run. It is released
/// when they return, or after a bound that only a `stop` waiting for the write
/// reaches, and it returns before the client resumes.
#[async_test]
async fn test_stop_then_pause_keeps_a_room_whose_first_write_is_in_flight() {
    let room_id = room_id!("!room:localhost");
    let first = event_id!("$first:localhost");
    let after = event_id!("$after-resume:localhost");

    let held = Arc::new(HeldWrite::default());
    *held.event_id.lock().unwrap() = Some(first.to_owned());
    let directory = tempfile::tempdir().unwrap();
    let event_cache_store = HoldingStore {
        inner: SqliteEventCacheStore::open(directory.path(), None)
            .await
            .unwrap()
            .into_event_cache_store(),
        held: held.clone(),
    };

    let server = MatrixMockServer::new().await;
    let client = server
        .client_builder()
        .on_builder(|builder| {
            builder.store_config(
                StoreConfig::new(CrossProcessLockConfig::SingleProcess)
                    .event_cache_store(event_cache_store),
            )
        })
        .build()
        .await;

    // Rooms go only to the room list connection, as from a real server.
    let pending = Arc::new(Mutex::new(vec![message(first)]));
    let pos = Arc::new(Mutex::new(0));
    Mock::given(SlidingSyncMatcher)
        .respond_with({
            let pending = pending.clone();
            move |request: &Request| {
                let partial: PartialSlidingSyncRequest = request.body_json().unwrap();
                let mut pos = pos.lock().unwrap();
                *pos += 1;
                let mut response = json!({ "txn_id": partial.txn_id, "pos": pos.to_string() });
                if partial.conn_id.as_deref() == Some("room-list") {
                    response["lists"] = json!({ "all_rooms": { "count": 1 } });
                    let events: Vec<Value> = pending.lock().unwrap().drain(..).collect();
                    if !events.is_empty() {
                        response["rooms"] = json!({
                            room_id: {
                                "name": "Room",
                                "initial": true,
                                "required_state": [
                                    {
                                        "type": "m.room.create", "state_key": "",
                                        "event_id": "$create:localhost", "sender": "@bob:localhost",
                                        "origin_server_ts": 0,
                                        "content": { "creator": "@bob:localhost", "room_version": "11" },
                                    },
                                    {
                                        "type": "m.room.member", "state_key": "@example:localhost",
                                        "event_id": "$member:localhost", "sender": "@example:localhost",
                                        "origin_server_ts": 0, "content": { "membership": "join" },
                                    },
                                ],
                                "timeline": events,
                                "limited": false,
                            }
                        });
                    }
                }
                ResponseTemplate::new(200)
                    .set_body_json(response)
                    .set_delay(Duration::from_millis(20))
            }
        })
        .mount(&server)
        .await;

    let sync_service = Arc::new(SyncService::builder(client.clone()).build().await.unwrap());
    sync_service.start().await;
    timeout(Duration::from_secs(10), held.held.notified())
        .await
        .expect("the event cache starts writing the room");

    let mut stop_then_pause = spawn({
        let sync_service = sync_service.clone();
        let client = client.clone();
        async move {
            sync_service.stop().await;
            client.pause().await.unwrap();
        }
    });
    let finished = timeout(Duration::from_secs(2), &mut stop_then_pause).await;
    held.released.notify_one();
    // The released write must reach the store before `resume` reopens it.
    timeout(Duration::from_secs(10), held.done.notified())
        .await
        .expect("the held write returns once released");
    match finished {
        Ok(finished) => finished.unwrap(),
        Err(_) => stop_then_pause.await.unwrap(),
    }

    client.resume().await.unwrap();
    pending.lock().unwrap().push(message(after));
    sync_service.start().await;

    let stored = timeout(Duration::from_secs(10), async {
        loop {
            let store = match client.event_cache_store().lock().await.unwrap() {
                MappedCrossProcessLockState::Clean(store)
                | MappedCrossProcessLockState::Dirty(store) => store,
            };
            let stored: Vec<OwnedEventId> = store
                .load_all_chunks(LinkedChunkId::Room(room_id))
                .await
                .unwrap()
                .into_iter()
                .filter_map(|chunk| match chunk.content {
                    ChunkContent::Items(events) => Some(events),
                    ChunkContent::Gap(_) => None,
                })
                .flatten()
                .filter_map(|event| event.event_id())
                .collect();
            if stored.iter().any(|id| id == after) {
                return stored;
            }
            drop(store);
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    })
    .await
    .expect("the room's event after the resume reaches the store");
    assert!(stored.iter().any(|id| id == first), "the held event was stored: {stored:?}");

    sync_service.stop().await;
}
