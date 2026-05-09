//! Block-allocator metadata: free list + refcounts + copy-on-write tracking.
//!
//! Mirrors `kvboost.cpu_paged.block_allocator.BlockAllocator` but owns only
//! the bookkeeping. Tensor pools and reads/writes stay on the Python side —
//! this struct hands back block ids and CoW decisions, and the caller does
//! the actual `torch.Tensor` indexing.

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

struct Inner {
    num_blocks: usize,
    free: Vec<usize>,        // stack of available block ids
    ref_count: HashMap<usize, u32>,
}

/// Decision returned by `ensure_writable`.
///
/// - `block_id == original` and `needs_copy == false` → write in place.
/// - `block_id != original` and `needs_copy == true`  → caller must copy
///   `original` → `block_id` across all layer pools, then write.
#[pyclass]
#[derive(Clone, Copy)]
pub struct CowDecision {
    #[pyo3(get)]
    pub block_id: usize,
    #[pyo3(get)]
    pub needs_copy: bool,
}

#[pyclass]
pub struct BlockAllocatorMeta {
    inner: Mutex<Inner>,
}

#[pymethods]
impl BlockAllocatorMeta {
    #[new]
    pub fn new(num_blocks: usize) -> Self {
        // Free list is a stack: pop from end. Initialize with reverse order
        // so allocations come out in ascending id order (matches Python's
        // `list(range(n)).pop()` behavior — pop returns highest id first;
        // we mirror exactly).
        let free: Vec<usize> = (0..num_blocks).collect();
        BlockAllocatorMeta {
            inner: Mutex::new(Inner {
                num_blocks,
                free,
                ref_count: HashMap::new(),
            }),
        }
    }

    /// Allocate `n` fresh physical blocks. Raises RuntimeError on OOM.
    pub fn allocate(&self, n: usize) -> PyResult<Vec<usize>> {
        let mut g = self.inner.lock().unwrap();
        if g.free.len() < n {
            return Err(PyRuntimeError::new_err(format!(
                "BlockAllocator OOM: requested {} blocks but only {} free (pool size {}).",
                n,
                g.free.len(),
                g.num_blocks
            )));
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let bid = g.free.pop().expect("checked above");
            g.ref_count.insert(bid, 1);
            out.push(bid);
        }
        Ok(out)
    }

    /// Decrement ref counts and return blocks to the free list when rc==0.
    pub fn free(&self, block_ids: Vec<usize>) {
        let mut g = self.inner.lock().unwrap();
        for bid in block_ids {
            let rc = g.ref_count.get(&bid).copied().unwrap_or(0);
            if rc <= 1 {
                g.ref_count.remove(&bid);
                g.free.push(bid);
            } else {
                g.ref_count.insert(bid, rc - 1);
            }
        }
    }

    /// Copy-on-write fork: bump ref counts, return the same id list.
    /// Caller must invoke `ensure_writable` before any write.
    pub fn fork(&self, block_ids: Vec<usize>) -> Vec<usize> {
        let mut g = self.inner.lock().unwrap();
        for &bid in &block_ids {
            let rc = g.ref_count.get(&bid).copied().unwrap_or(1);
            g.ref_count.insert(bid, rc + 1);
        }
        block_ids
    }

    /// Decide whether the caller can write to `block_id` directly, or whether
    /// it must allocate a fresh block and copy first.
    ///
    /// If a copy is needed and we OOM during the fresh-block reservation,
    /// raises RuntimeError (matches the Python original's behavior).
    pub fn ensure_writable(&self, block_id: usize) -> PyResult<CowDecision> {
        let mut g = self.inner.lock().unwrap();
        let rc = g.ref_count.get(&block_id).copied().unwrap_or(1);
        if rc <= 1 {
            return Ok(CowDecision {
                block_id,
                needs_copy: false,
            });
        }
        if g.free.is_empty() {
            return Err(PyRuntimeError::new_err(
                "BlockAllocator OOM during copy-on-write.",
            ));
        }
        let new_id = g.free.pop().expect("checked above");
        g.ref_count.insert(new_id, 1);
        g.ref_count.insert(block_id, rc - 1);
        Ok(CowDecision {
            block_id: new_id,
            needs_copy: true,
        })
    }

    // ── Stats ────────────────────────────────────────────────────────────────

    #[getter]
    pub fn num_blocks(&self) -> usize {
        self.inner.lock().unwrap().num_blocks
    }

    #[getter]
    pub fn free_blocks(&self) -> usize {
        self.inner.lock().unwrap().free.len()
    }

    #[getter]
    pub fn used_blocks(&self) -> usize {
        let g = self.inner.lock().unwrap();
        g.num_blocks - g.free.len()
    }

    pub fn utilization(&self) -> f64 {
        let g = self.inner.lock().unwrap();
        (g.num_blocks - g.free.len()) as f64 / g.num_blocks.max(1) as f64
    }

    /// Read-only ref-count snapshot (mostly for tests/parity).
    pub fn ref_count_snapshot(&self) -> HashMap<usize, u32> {
        self.inner.lock().unwrap().ref_count.clone()
    }

    /// Read-only free-list snapshot (order matters for parity).
    pub fn free_snapshot(&self) -> Vec<usize> {
        self.inner.lock().unwrap().free.clone()
    }
}
