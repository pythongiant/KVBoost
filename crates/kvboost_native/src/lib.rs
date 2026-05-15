//! kvboost_native — Rust hot-path components for kvboost.
//!
//! Currently exposes:
//!   - BlockAllocatorMeta : free-list + refcount bookkeeping for the CPU
//!     paged-attention block pool. Tensor I/O stays in Python.

use pyo3::prelude::*;

mod block_allocator;
mod streaming_sched;

#[pymodule]
fn kvboost_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<block_allocator::BlockAllocatorMeta>()?;
    m.add_class::<block_allocator::CowDecision>()?;
    m.add_class::<streaming_sched::StreamingSchedulerBackend>()?;
    m.add_class::<streaming_sched::SlotAssignment>()?;
    Ok(())
}
