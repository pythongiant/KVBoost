//! Streaming scheduler bookkeeping — slot/event tracking for layer streaming.
//!
//! Mirrors the pattern of [`block_allocator::BlockAllocatorMeta`]: this struct
//! owns *only* metadata. CUDA event creation / recording / waiting stays in
//! Python (where `torch.cuda.Event` lives). The Rust side just tracks which
//! layer occupies which slot, and which opaque event handle (a `u64` chosen
//! by the Python caller, typically `id(event)`) gates each slot's reuse.

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

struct Inner {
    num_slots: usize,
    /// slot_id -> currently-resident layer_idx (None means slot is empty).
    slot_to_layer: Vec<Option<usize>>,
    /// layer_idx -> slot currently holding its weights.
    layer_to_slot: HashMap<usize, usize>,
    /// slot_id -> opaque transfer-completion event handle (xfer_done).
    slot_xfer_event: Vec<Option<u64>>,
    /// slot_id -> opaque compute-completion event handle (compute_done).
    slot_compute_event: Vec<Option<u64>>,
    /// Monotonic counter for synthetic event ids when the caller doesn't
    /// supply one. Real CUDA event objects live in Python.
    next_synthetic_event: u64,
}

/// Result of `assign_slot`: which slot the layer was placed in, and the
/// opaque event the caller must `wait_event` on before reusing that slot's
/// staging memory (i.e. the previous occupant's compute_done event).
#[pyclass]
#[derive(Clone, Copy)]
pub struct SlotAssignment {
    #[pyo3(get)]
    pub slot_id: usize,
    #[pyo3(get)]
    pub evicted_layer: Option<usize>,
    /// The compute_done event of the previous occupant, or None if the slot
    /// was empty / its compute already retired.
    #[pyo3(get)]
    pub wait_event: Option<u64>,
}

#[pyclass]
pub struct StreamingSchedulerBackend {
    inner: Mutex<Inner>,
}

#[pymethods]
impl StreamingSchedulerBackend {
    #[new]
    pub fn new(num_slots: usize) -> PyResult<Self> {
        if num_slots == 0 {
            return Err(PyValueError::new_err("num_slots must be >= 1"));
        }
        Ok(StreamingSchedulerBackend {
            inner: Mutex::new(Inner {
                num_slots,
                slot_to_layer: vec![None; num_slots],
                layer_to_slot: HashMap::new(),
                slot_xfer_event: vec![None; num_slots],
                slot_compute_event: vec![None; num_slots],
                next_synthetic_event: 1,
            }),
        })
    }

    /// Allocate a fresh synthetic event handle.
    ///
    /// The caller may either use this id as a key for its own `torch.cuda.Event`
    /// map, or ignore it and pass `id(event)` of a real Python event as the
    /// `event_handle` parameter elsewhere — both are opaque u64s here.
    pub fn allocate_event(&self) -> u64 {
        let mut g = self.inner.lock().unwrap();
        let id = g.next_synthetic_event;
        g.next_synthetic_event = g.next_synthetic_event.wrapping_add(1);
        id
    }

    /// Pick a slot for `layer_idx` using round-robin over `layer_idx % num_slots`.
    /// Returns the slot id plus any previous occupant whose compute must drain
    /// before the new transfer can land.
    pub fn assign_slot(&self, layer_idx: usize) -> PyResult<SlotAssignment> {
        let mut g = self.inner.lock().unwrap();
        if g.num_slots == 0 {
            return Err(PyRuntimeError::new_err("scheduler has zero slots"));
        }
        let slot_id = layer_idx % g.num_slots;
        let evicted_layer = g.slot_to_layer[slot_id];
        let wait_event = g.slot_compute_event[slot_id];

        if let Some(prev_layer) = evicted_layer {
            g.layer_to_slot.remove(&prev_layer);
        }
        g.slot_to_layer[slot_id] = Some(layer_idx);
        g.layer_to_slot.insert(layer_idx, slot_id);
        // The new transfer hasn't been recorded yet; clear stale events for
        // this slot — the caller will publish them via record_*_event below.
        g.slot_xfer_event[slot_id] = None;
        g.slot_compute_event[slot_id] = None;

        Ok(SlotAssignment {
            slot_id,
            evicted_layer,
            wait_event,
        })
    }

    /// Record that the transfer-done event for the layer occupying `slot_id`
    /// has been issued by the caller on the transfer stream.
    pub fn record_transfer_event(&self, slot_id: usize, event_handle: u64) -> PyResult<()> {
        let mut g = self.inner.lock().unwrap();
        check_slot(&g, slot_id)?;
        g.slot_xfer_event[slot_id] = Some(event_handle);
        Ok(())
    }

    /// Record that the compute-done event for the layer occupying `slot_id`
    /// has been issued by the caller on the compute stream.
    pub fn record_compute_event(&self, slot_id: usize, event_handle: u64) -> PyResult<()> {
        let mut g = self.inner.lock().unwrap();
        check_slot(&g, slot_id)?;
        g.slot_compute_event[slot_id] = Some(event_handle);
        Ok(())
    }

    /// Forward-lookup: which slot currently holds `layer_idx`?
    pub fn get_slot_for_layer(&self, layer_idx: usize) -> Option<usize> {
        self.inner.lock().unwrap().layer_to_slot.get(&layer_idx).copied()
    }

    /// Reverse-lookup: which layer is in `slot_id`?
    pub fn get_layer_for_slot(&self, slot_id: usize) -> PyResult<Option<usize>> {
        let g = self.inner.lock().unwrap();
        check_slot(&g, slot_id)?;
        Ok(g.slot_to_layer[slot_id])
    }

    /// Get the transfer-done event for a slot, if recorded.
    pub fn get_transfer_event(&self, slot_id: usize) -> PyResult<Option<u64>> {
        let g = self.inner.lock().unwrap();
        check_slot(&g, slot_id)?;
        Ok(g.slot_xfer_event[slot_id])
    }

    /// Get the compute-done event for a slot, if recorded.
    pub fn get_compute_event(&self, slot_id: usize) -> PyResult<Option<u64>> {
        let g = self.inner.lock().unwrap();
        check_slot(&g, slot_id)?;
        Ok(g.slot_compute_event[slot_id])
    }

    /// Reset all slot state. Useful between forward passes.
    pub fn reset(&self) {
        let mut g = self.inner.lock().unwrap();
        for s in g.slot_to_layer.iter_mut() {
            *s = None;
        }
        for s in g.slot_xfer_event.iter_mut() {
            *s = None;
        }
        for s in g.slot_compute_event.iter_mut() {
            *s = None;
        }
        g.layer_to_slot.clear();
    }

    // ── Stats ────────────────────────────────────────────────────────────────

    #[getter]
    pub fn num_slots(&self) -> usize {
        self.inner.lock().unwrap().num_slots
    }

    /// Snapshot of (slot_id, occupying layer) for debugging.
    pub fn slot_snapshot(&self) -> Vec<(usize, Option<usize>)> {
        let g = self.inner.lock().unwrap();
        g.slot_to_layer
            .iter()
            .enumerate()
            .map(|(i, v)| (i, *v))
            .collect()
    }
}

fn check_slot(g: &Inner, slot_id: usize) -> PyResult<()> {
    if slot_id >= g.num_slots {
        return Err(PyValueError::new_err(format!(
            "slot_id {} out of range (num_slots={})",
            slot_id, g.num_slots
        )));
    }
    Ok(())
}
