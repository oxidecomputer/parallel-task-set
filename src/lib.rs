// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::future::Future;
use std::sync::Arc;
use tokio::sync::{Semaphore, TryAcquireError};
use tokio::task::JoinSet;

/// The default number of parallel tasks used by [ParallelTaskSet].
pub const DEFAULT_MAX_PARALLELISM: usize = 16;

/// A collection of tokio tasks which execute in parallel on distinct tokio
/// tasks, up to a user-specified maximum amount of parallelism.
///
/// This parallelism is achieved by spawning tasks on a [JoinSet],
/// and may be further limited by the underlying machine's ability
/// to execute many tokio tasks.
///
/// # Why not just use FuturesUnordered?
///
/// FuturesUnordered can execute any number of futures concurrently,
/// but makes no attempt to execute them in parallel (assuming the underlying
/// futures are not themselves spawning additional tasks).
///
/// # Why not just use a JoinSet?
///
/// The tokio [JoinSet] has not limit on the "maximum number of tasks".
/// Given a bursty workload, it's possible to spawn an enormous number
/// of tasks, which may not be desirable.
///
/// Although [ParallelTaskSet] uses a [JoinSet] internally, it also
/// respects a parallism capacity.
pub struct ParallelTaskSet<T> {
    semaphore: Arc<Semaphore>,
    set: JoinSet<T>,
}

impl<T: 'static + Send> Default for ParallelTaskSet<T> {
    fn default() -> Self {
        ParallelTaskSet::new()
    }
}

impl<T: 'static + Send> ParallelTaskSet<T> {
    /// Creates a new [ParallelTaskSet], with [DEFAULT_MAX_PARALLELISM] as the
    /// maximum number of tasks to run in parallel.
    ///
    /// If a different amount of parallism is desired, refer to:
    /// [Self::new_with_parallelism].
    pub fn new() -> ParallelTaskSet<T> {
        Self::new_with_parallelism(DEFAULT_MAX_PARALLELISM)
    }

    /// Creates a new [ParallelTaskSet], with `max_parallism` as the
    /// maximum number of tasks to run in parallel.
    pub fn new_with_parallelism(max_parallism: usize) -> ParallelTaskSet<T> {
        let semaphore = Arc::new(Semaphore::new(max_parallism));
        let set = JoinSet::new();

        Self { semaphore, set }
    }

    /// Spawn the provided `next_future`, potentially returning the output of a
    /// previously-spawned task.
    ///
    /// If this `ParallelTaskSet` is at its `max_parallelism`, this method will
    /// first wait until a previously-spawned task is complete, so that no more
    /// than `max_parallelism` tasks are active. If this occurs, this method
    /// will return that task's output. Otherwise, if waiting for an existing
    /// task is not necessary, this method returns `None`, even if some tasks
    /// are ready.
    ///
    /// Note that, *unlike* [`join_next()`], this method returning `None` does
    /// NOT indicate that the set is empty, just that it was not necessary to
    /// wait for a previous task to complete in order to spawn the existing one.
    /// Therefore, callers wishing to ensure all tasks spawned on this
    /// [`ParallelTaskSet`] have completed should consider using [`join_next()`]
    /// once all tasks have been spawned.
    ///
    /// [`join_next()`]: Self::join_next
    #[must_use = "ParallelTaskSet::spawn returns a task output once \
        `max_parallelism` tasks are spawned"]
    pub async fn spawn<F>(&mut self, future: F) -> JoinedTasks<'_, T>
    where
        F: Future<Output = T> + Send + 'static,
    {
        let (permit, output) =
            match Arc::clone(&self.semaphore).try_acquire_owned() {
                Ok(permit) => (permit, None),
                Err(TryAcquireError::Closed) => {
                    unreachable!("we never close the semaphore")
                }
                Err(TryAcquireError::NoPermits) => {
                    let joined = self.join_next().await;
                    let permit = Arc::clone(&self.semaphore)
                        .acquire_owned()
                        .await
                        .expect("we never close the semaphore");
                    (permit, joined)
                }
            };

        self.set.spawn(async move {
            let output = future.await;
            drop(permit);
            output
        });

        JoinedTasks { next: output, set: self }
    }

    /// Waits for the next task to complete and return its output.
    ///
    /// # Panics
    ///
    /// This method panics if the task returns a JoinError
    pub async fn join_next(&mut self) -> Option<T> {
        self.set.join_next().await.map(|r| r.expect("Failed to join task"))
    }

    /// Returns the output of a task that has completed, if there are any such
    /// tasks.
    ///
    /// Unlike [`join_next`], this method does not wait for a task to complete
    /// if no tasks have completed.
    ///
    /// Returns `None` if no tasks have completed, or if the [`ParallelTaskSet`]
    /// is empty.
    ///
    /// # Panics
    ///
    /// This method panics if the next completed task has returned a
    /// [`tokio::task::JoinError`].
    pub fn try_join_next(&mut self) -> Option<T> {
        self.set.try_join_next().map(|r| r.expect("Joined task has failed"))
    }

    /// Wait for all remaining tasks to complete and return their output.
    ///
    /// Note that, unlike [`JoinSet`], this **cannot** be used to collect all of
    /// the output at once, because [`spawn`][Self::spawn] will return outputs
    /// once `max_parallelism` tasks are added to the set.
    ///
    /// # Panics
    ///
    /// This method panics any of the tasks return a JoinError
    pub async fn join_remaining(self) -> Vec<T> {
        self.set.join_all().await
    }
}

#[must_use = "dropping a `JoinedTasks` without consuming the output of the \
    completed tasks will reduce the available capacity of the task set until \
    those tasks' output is consumed"]
pub struct JoinedTasks<'set, T> {
    next: Option<T>,
    set: &'set mut ParallelTaskSet<T>,
}

impl<'set, T> JoinedTasks<'set, T>
where
    T: Send + 'static,
{
    /// Discard the output of all currently-completed tasks on the
    /// [`ParallelTaskSet`], freeing up capacity for additional tasks to be
    /// spawned.
    ///
    /// This is particularly useful when the output type of the futures spawned
    /// on the [`ParallelTaskSet`] is `()`, and the caller does not wish to
    /// explicitly loop over this iterator to consume a whole bunch of nothing.
    pub fn discard(self) {
        for _ in self {}
    }
}

impl<'set, T> Iterator for JoinedTasks<'set, T>
where
    T: Send + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.take().or_else(|| self.set.try_join_next())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::Rng;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    #[tokio::test]
    async fn test_spawn_many() {
        let count = Arc::new(AtomicUsize::new(0));

        let task_limit = 16;
        let mut set = ParallelTaskSet::new_with_parallelism(task_limit);
        let mut i = 0;

        let mut check_watermark = |watermark: usize| {
            println!("task {i} saw {watermark} concurrent tasks");

            assert!(
                watermark <= task_limit,
                "Observed simultaneous task execution of {watermark} tasks on the {i}-th worker"
            );

            i += 1;
        };

        for _ in 0..task_limit * 10 {
            let completed = set
                .spawn({
                    let count = count.clone();
                    async move {
                        // How many tasks - including our own - are running right
                        // now?
                        let watermark =
                            count.fetch_add(1, Ordering::SeqCst) + 1;

                        // The tasks should all execute for a short but variable
                        // amount of time.
                        let duration_ms = rand::rng().random_range(0..10);
                        tokio::time::sleep(tokio::time::Duration::from_millis(
                            duration_ms,
                        ))
                        .await;

                        count.fetch_sub(1, Ordering::SeqCst);

                        watermark
                    }
                })
                .await;
            for task in completed {
                check_watermark(task);
            }
        }

        let watermarks = set.join_remaining().await;
        for watermark in watermarks {
            check_watermark(watermark);
        }
    }
}
