// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::Arc;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::task::JoinError;
use tokio::task::JoinSet;

/// The default number of parallel tasks used by [ParallelTaskSet].
pub const DEFAULT_MAX_PARALLELISM: NonZeroUsize = match NonZeroUsize::new(16) {
    Some(n) => n,
    None => unreachable!(), // 16 != 0 (source: that's just how numbers work)
};

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
///
/// # Limiting Parallelism
///
/// A `ParallelTaskSet` holds at most `max_parallelism` tasks at a time. This
/// limit is intended to avoid resource exhaustion from large numbers of tasks.
/// For this reason, the limit applies not only to the number of tasks currently
/// *executing* on the set, but also to the number of *completed* tasks whose
/// output has not yet been consumed. This is because a task which has completed
/// but not yet joined retains its heap allocation (and does not drop any RAII
/// resources it may have acquired) until its output is consumed.
///
/// Therefore, code which wishes to spawn a new task on the [`ParallelTaskSet`]
/// must consume the output of completed tasks as part of the `spawn` operation.
/// This results in an API that differs substantially from Tokio's [`JoinSet`],
/// as we are about to discuss.
///
/// # Usage
///
/// Unlike [`JoinSet::spawn()`], [`ParallelTaskSet::spawn()`] is an async
/// function, and returns the outputs of previously-spawned tasks. When the set
/// is at capacity, calling [`spawn()`] must wait for a task to complete, and
/// the caller must take its output to free space for the new task. In this
/// case, [`spawn()`] returns that output, along with those of any other tasks
/// that have completed, as a [`CompletedTasks`] iterator
///
/// An output returned by [`spawn()`] is not returned again by [`join_next()`] or
/// [`join_remaining()`]. This means that code that ignores the returned iterator
/// from a call to [`spawn()`] and collects outputs at the end (as one might with
/// a Tokio [`JoinSet`]) will lose all but the last few outputs.
///
/// Instead, spawn tasks in a loop, handling outputs each time a new task is
/// spawned. Once no more tasks must be spawned, collect any remaining tasks as
/// they complete, using [`join_remaining()`] or [`join_next()`]. For example:
///
/// ```
/// # #[tokio::main(flavor = "current_thread")]
/// # async fn main() {
/// use parallel_task_set::ParallelTaskSet;
///
/// let limit = NonZeroUsize::new(4).expect("4 != 0");
/// let mut tasks = ParallelTaskSet::new_with_parallelism(limit);
/// let mut outputs = Vec::new();
///
/// for i in 0..100 {
///     // Once the set holds four tasks, `spawn` waits for one of them to
///     // complete before spawning the next, and returns its output. The return
///     // value from `spawn` is an iterator which may contain any number of
///     // completed task outputs.
///     outputs.extend(tasks.spawn(async move { i * 2 }).await);
/// }
///
/// // `join_remaining` returns only the outputs of any tasks which `spawn` has
/// // not already returned.
/// outputs.extend(tasks.join_remaining().await);
///
/// // Tasks may complete in any order.
/// outputs.sort();
/// assert_eq!(outputs, (0..100).map(|i| i * 2).collect::<Vec<_>>());
/// # }
/// ```
///
/// If the tasks' outputs are not needed (for instance, because `T` is `()`),
/// use [`CompletedTasks::discard`] to throw them away, draining the
/// `CompletedTasks` iterator. This ensures that capacity is freed for remaining
/// tasks, even when the caller does not need to collect the output of those
/// tasks.
///
/// [`spawn()`]: ParallelTaskSet::spawn
/// [`join_next()`]: ParallelTaskSet::join_next
/// [`join_remaining()`]: ParallelTaskSet::join_remaining
pub struct ParallelTaskSet<T> {
    semaphore: Arc<Semaphore>,
    set: JoinSet<(OwnedSemaphorePermit, T)>,
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
    pub fn new_with_parallelism(
        max_parallism: NonZeroUsize,
    ) -> ParallelTaskSet<T> {
        let semaphore = Arc::new(Semaphore::new(max_parallism.get()));
        let set = JoinSet::new();

        Self { semaphore, set }
    }

    /// Spawn the provided `future`, potentially returning the outputs of
    /// previously-spawned tasks.
    ///
    /// If this `ParallelTaskSet` is at its [`max_parallelism`], this method
    /// will first wait until a previously-spawned task is complete, so that no
    /// more than `max_parallelism` tasks are active. If this occurs, the
    /// returned [`CompletedTasks`] iterator will yield that task's output
    /// first. The iterator then yields the outputs of any other tasks that have
    /// completed, without waiting for tasks which are still running.
    ///
    /// Note that, *unlike* [`join_next()`], this iterator returning `None` does
    /// NOT indicate that the set is empty, just that no other tasks have
    /// completed yet. Therefore, callers wishing to ensure all tasks spawned on
    /// this [`ParallelTaskSet`] have completed should consider using
    /// [`join_next()`] or [`join_remaining()`] once all tasks have been
    /// spawned.
    ///
    /// Dropping the iterator without exhausting it does not lose any outputs,
    /// *except* the output of the task that this method waited for, which has
    /// already been removed from the set. See [`CompletedTasks`] for details.
    ///
    /// # Cancel safety
    ///
    /// If the future returned by this method is dropped before it completes,
    /// `future` is dropped without having been spawned. No previously-spawned
    /// task's output is lost: once this method has taken a completed task's
    /// output, it returns without awaiting anything further.
    ///
    /// # Panics
    ///
    /// This method panics if it must await the completion of a currently
    /// running task, and that task returns a [`tokio::task::JoinError`].
    ///
    /// [`max_parallelism`]: ParallelTaskSet#limiting-parallelism
    /// [`join_next()`]: Self::join_next
    /// [`join_remaining()`]: Self::join_remaining
    #[must_use = "`ParallelTaskSet::spawn` returns the outputs of completed \
        tasks, which will not be returned again by `join_next()` or \
        `join_remaining()`. outputs are, therefore, lost if the returned \
        iterator is not consumed."]
    pub async fn spawn<F>(&mut self, future: F) -> CompletedTasks<'_, T>
    where
        F: Future<Output = T> + Send + 'static,
    {
        let (permit, output) = match Arc::clone(&self.semaphore)
            .try_acquire_owned()
        {
            Ok(permit) => (permit, None),
            Err(TryAcquireError::Closed) => {
                unreachable!("we never close the semaphore")
            }
            Err(TryAcquireError::NoPermits) => {
                let joined =
                    unwrap_join_result(self.set.join_next().await.transpose());
                match joined {
                    // If another task has completed, let's reuse its
                    // permit, avoiding a few atomic operations (to drop the
                    // permit, which increments the semaphore's permit count
                    // and decrements the `Arc<Semaphore>`'s ref count, and
                    // then to acquire a new permit, which does the
                    // inverse...).
                    Some((permit, output)) => (permit, Some(output)),
                    // It would be strange to end up here: the call to
                    // `try_acquire_owned()` should not return `NoPermits`
                    // unless there was a joined task to consume, but
                    // somehow, there wasn't one. Nonetheless, let's handle
                    // that gracefully by acquiring a new one.
                    None => {
                        let permit = Arc::clone(&self.semaphore)
                            .acquire_owned()
                            .await
                            .expect("we never close the semaphore");
                        (permit, None)
                    }
                }
            }
        };

        self.set.spawn(async move {
            let output = future.await;
            (permit, output)
        });

        CompletedTasks { next: output, set: self }
    }

    /// Waits for the next task to complete and return its output.
    ///
    /// # Panics
    ///
    /// This method panics if the task returns a JoinError
    pub async fn join_next(&mut self) -> Option<T> {
        self.set.join_next().await.map(|r| {
            let (_permit, output) = unwrap_join_result(r);
            output
        })
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
    ///
    /// [`join_next`]: Self::join_next
    pub fn try_join_next(&mut self) -> Option<T> {
        self.set.try_join_next().map(|r| {
            let (_permit, output) = unwrap_join_result(r);
            output
        })
    }

    /// Wait for all remaining tasks to complete and return their output.
    ///
    /// Note that, unlike the similarly named [`tokio::task::JoinSet`] method,
    /// this **cannot** be used to collect all of the output at once, because
    /// [`spawn`][Self::spawn] returns the outputs of tasks as they complete.
    ///
    /// # Panics
    ///
    /// This method panics any of the tasks return a JoinError
    pub async fn join_remaining(mut self) -> Vec<T> {
        // N.B. that this does not use `JoinSet::join_all`, because that
        // function returns a `Vec` of each task's output. If we were to use
        // that here, we would need to drain that vec and then allocate a new
        // one, just to get rid of the permits that each task outputs alongside
        // their actual output. By looping ourselves, we avoid the superfluous
        // allocation.
        let mut output = Vec::with_capacity(self.set.len());
        while let Some(task) = self.join_next().await {
            output.push(task);
        }
        output
    }
}

/// An iterator over the outputs of completed tasks, returned by
/// [`ParallelTaskSet::spawn`].
///
/// If the call to `spawn` had to wait for a task to complete before spawning a
/// new one, the first item in this iterator is that task's output. Subsequent
/// items are the outputs of any other tasks in the set which have completed.
///
/// Because tasks may complete concurrently while this iterator exists, it is
/// not fused. If this iterator returns `None`, it may return `Some` again if
/// additional tasks complete. However, because this iterator mutably borrows
/// the [`ParallelTaskSet`], no new tasks can be spawned on the set until it is
/// dropped. Therefore, the iterator will never return more than the configured
/// [`max_parallelism`] of the [`ParallelTaskSet`].
///
/// # Dropping
///
/// When this iterator is dropped, any tasks which have completed but whose
/// output has not been returned by the iterator remain in the
/// [`ParallelTaskSet`]. These tasks continue to count against the
/// `ParallelTaskSet`'s [`max_parallelism`], and will be returned by a later
/// call to [`spawn`], [`join_next`](ParallelTaskSet::join_next), or
/// [`join_remaining`](ParallelTaskSet::join_remaining).
///
/// However, if the call to [`spawn`] awaited the completion of an existing task
/// because the `ParallelTaskSet` was at capacity, that task's output for has
/// already been taken from the set and is held by the iterator. Therefore,
/// dropping the iterator before taking its first item discards that output.
///
/// For both of these reasons, this type is `must_use`. Callers should always
/// consume the iterator to retrieve any task output that would otherwise be
/// lost, *and* to ensure that any resources held by
/// completed-but-not-yet-consumed tasks are released eagerly. If you do not
/// need the output values (i.e. because they are `()`), use
/// [`discard`](Self::discard) and to free the capacity occupied by completed
/// tasks.
///
/// [`max_parallelism`]: ParallelTaskSet#limiting-parallelism
/// [`spawn`]: ParallelTaskSet::spawn
#[must_use = "dropping a `CompletedTasks` without consuming it may discard \
    the output of a task that had to be awaited to free capacity for a new one \
    to be spawned"]
pub struct CompletedTasks<'set, T> {
    next: Option<T>,
    set: &'set mut ParallelTaskSet<T>,
}

impl<'set, T> CompletedTasks<'set, T>
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

impl<'set, T> Iterator for CompletedTasks<'set, T>
where
    T: Send + 'static,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.next.take().or_else(|| self.set.try_join_next())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // The lower bound is 1 if we have at least one output to return.
        let lower = self.next.is_some() as usize;
        // The upper bound is the number of tasks currently spawned on the
        // joinset, plus `self.next`, whose task has already been removed from
        // the joinset. Because this iterator holds an `&mut` reference to the
        // JoinSet, we know no new tasks will be spawned while this iterator
        // exists.
        let upper = Some(self.set.set.len() + lower);
        (lower, upper)
    }
}

fn unwrap_join_result<T>(result: Result<T, JoinError>) -> T {
    match result {
        Ok(output) => output,
        Err(err) => std::panic::resume_unwind(err.into_panic()),
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

        let task_limit = NonZeroUsize::new(16).expect("16 != 0");
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

    /// Tests that tests count against the parallelism limit until their outputs
    /// are consumed. This is a regression test for the behavioral change in PR
    /// #1.
    #[tokio::test]
    async fn test_capacity_not_released_until_drained() {
        const TASKS: usize = 100;
        let limit = NonZeroUsize::new(4).expect("4 != 0");
        let mut set = ParallelTaskSet::new_with_parallelism(limit);
        let mut outputs = Vec::with_capacity(TASKS);

        for i in 0..TASKS {
            outputs.extend(set.spawn(async move { i }).await);
            assert!(
                set.set.len() <= limit,
                "after spawning task {i}, the ParallelTaskSet holds {} tasks, \
                 exceeding the desired limit of {limit}",
                set.set.len(),
            );
            // Let the spawned tasks run to completion, so that the set holds
            // completed tasks whose outputs have not yet been consumed.
            tokio::task::yield_now().await;
        }

        let remaining = set.join_remaining().await;
        assert!(
            remaining.len() <= limit,
            "join_remaining returned {} outputs, but at most {limit} tasks \
             should remain in the ParallelTaskSet",
            remaining.len(),
        );
        outputs.extend(remaining);

        outputs.sort_unstable();
        assert_eq!(
            outputs,
            (0..TASKS).collect::<Vec<_>>(),
            "every task's output must be returned exactly once",
        );
    }
}
