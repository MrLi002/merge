# E-RAFT source provenance

Vendored from [uzh-rpg/E-RAFT](https://github.com/uzh-rpg/E-RAFT/tree/c58ce0524ea0ebfa9849991caafb547f44fe9bfd), commit `c58ce0524ea0ebfa9849991caafb547f44fe9bfd`.
Files: `model/{eraft,extractor,update,corr,utils}.py`; MIT license retained in `LICENSE`.
`padding.py` adapts upstream `utils/image_utils.py::ImagePadder`.

Local compatibility changes: package relative imports, explicit meshgrid indexing,
removal of unused SciPy imports, PyTorch 2.5 `torch.amp.autocast` API (mixed
precision still disabled), input/iteration validation, per-forward padding
state, minimum 128x128 padded size to keep four-level correlations finite.
Feature/context encoders, image2 context convention, correlation ordering,
GRU, learned convex upsampling, parameter names and tensor shapes are unchanged.
The first forward result is 1/8-resolution displacement; the second is a list of
full-resolution predictions. Inference uses the last list entry.
