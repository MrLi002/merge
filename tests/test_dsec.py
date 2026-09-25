"""DSEC format and interval checks using explicitly synthetic local fixtures."""
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch

from ev6d.dsec import (DSECFlowDataset, create_synthetic_dsec_fixture,
                       read_flow_png, read_flow_timestamps, slice_events,
                       validate_split)


@pytest.fixture
def fixture_root(tmp_path):
    return create_synthetic_dsec_fixture(tmp_path/'中文路径'/'dsec', size=32, samples=3)


def test_synthetic_dsec_loader_unicode_units_validity_and_numeric_order(fixture_root):
    folder = fixture_root/'synthetic_train'/'flow'/'forward'
    for old, new in zip(sorted(folder.glob('*.png')), ('1.png', '2.png', '10.png')):
        old.rename(folder/new)
    dataset = DSECFlowDataset(fixture_root, ['synthetic_train'], num_bins=5)
    assert [int(row[1].stem) for row in dataset.records] == [1, 2, 10]
    sample = dataset[0]
    assert sample['old'].shape == (5, 32, 32)
    assert sample['old'].dtype == torch.float32 and sample['usable_events']
    assert sample['t_start'] == pytest.approx(1.1)
    assert sample['t_end'] == pytest.approx(1.2)
    assert torch.all(sample['flow'][0][sample['valid']] == 1)
    assert torch.all(sample['flow'][1] == 0)
    assert not sample['valid'][:, -2:].any()
    assert dataset.summary()['flow_unit'] == 'pixel/interval'


@pytest.mark.parametrize('content', [
    '1000000,1100000\n1100000,1200000',
    '# start_us,end_us\n1000000 1100000\n1100000 1200000',
    '\ufeff1000000, 1100000 # interval\n1100000,1200000\n',
])
def test_integer_timestamps_with_delimiters_and_comments(tmp_path, content):
    path = tmp_path/'time.txt'
    path.write_text(content, encoding='utf-8')
    np.testing.assert_array_equal(read_flow_timestamps(path), [[1000000, 1100000], [1100000, 1200000]])


@pytest.mark.parametrize('content', [
    '', '# no data', '1.5,2', '1e3,2000', 'nan,2000', '-1,2000',
    '2,2', '2,1', '1,3\n1,4', '1,3,4', '1,999999999999999999999999999',
])
def test_invalid_timestamps_rejected_without_integer_truncation(tmp_path, content):
    path = tmp_path/'time.txt'
    path.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        read_flow_timestamps(path)


def test_flow_png_channel_order_fractional_decode_and_invalid_mask(tmp_path):
    path = tmp_path/'中文.png'
    rgb = np.array([[[32768+160, 32768-64, 1], [60000, 60000, 0]]], dtype=np.uint16)
    cv2.imencode('.png', rgb[..., ::-1])[1].tofile(path)
    flow, valid = read_flow_png(path)
    np.testing.assert_allclose(flow[:, 0, 0], [1.25, -.5])
    assert valid.tolist() == [[True, False]]
    np.testing.assert_array_equal(flow[:, 0, 1], [0, 0])
    rgb[0, 0, 2] = 255
    cv2.imencode('.png', rgb[..., ::-1])[1].tofile(path)
    with pytest.raises(ValueError, match='0/1'):
        read_flow_png(path)
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='uint16'):
        read_flow_png(path)


def test_event_slicing_half_open_rectification_and_offset(tmp_path):
    path = tmp_path/'events.h5'
    times = np.array([0, 499, 500, 999, 1000, 1500, 2000], dtype=np.int64)
    with h5py.File(path, 'w') as handle:
        for name, values in {'t': times, 'x': np.ones(7, dtype=int),
                             'y': np.zeros(7, dtype=int), 'p': np.arange(7) % 2}.items():
            handle[f'events/{name}'] = values
        handle['t_offset'] = 1000000
        handle['ms_to_idx'] = np.searchsorted(times, [0, 1000, 2000, 3000])
    rectify_map = np.array([[[0., 0.], [1.25, .5]]], dtype=np.float32)
    with h5py.File(path, 'r') as handle:
        first = slice_events(handle, 1000000, 1001000, rectify_map)
        second = slice_events(handle, 1001000, 1002000, rectify_map)
        np.testing.assert_allclose(first[:, 0], [1., 1.000499, 1.0005, 1.000999])
        np.testing.assert_allclose(second[:, 0], [1.001, 1.0015])
        np.testing.assert_allclose(first[:, 1:], np.c_[np.full(4, 1.25), np.full(4, .5), [0, 1, 0, 1]])
        assert len(slice_events(handle, 1001100, 1001200, rectify_map)) == 0
    with h5py.File(path, 'r+') as handle:
        handle['ms_to_idx'][1] = 3
    with h5py.File(path, 'r') as handle, pytest.raises(ValueError, match='inconsistent'):
        slice_events(handle, 1001000, 1002000, rectify_map)


@pytest.mark.parametrize('problem', ['float_coordinates', 'bad_lengths', 'missing_ms', 'bad_ms', 'bad_polarity'])
def test_raw_hdf5_problems_are_explicit(fixture_root, problem):
    path = fixture_root/'synthetic_train'/'events'/'left'/'events.h5'
    with h5py.File(path, 'r+') as handle:
        if problem in ('float_coordinates', 'bad_lengths'):
            values = np.asarray(handle['events/x'])
            del handle['events/x']
            handle['events/x'] = values.astype(float) if problem == 'float_coordinates' else values[:-1]
        elif problem == 'missing_ms':
            del handle['ms_to_idx']
        elif problem == 'bad_ms':
            handle['ms_to_idx'][0] = -1
        else:
            handle['events/p'][:] = 2
    with pytest.raises(ValueError):
        DSECFlowDataset(fixture_root, ['synthetic_train'])[0]


def test_empty_cropped_support_skips_supervision(fixture_root):
    path = fixture_root/'synthetic_train'/'events'/'left'/'events.h5'
    with h5py.File(path, 'r+') as handle:
        handle['events/x'][:] = 0
        handle['events/y'][:] = 0
    sample = DSECFlowDataset(fixture_root, ['synthetic_train'], crop_size=[8, 8])[0]
    assert not sample['old'].any() and not sample['new'].any()
    assert not sample['usable_events'] and not sample['valid'].any()


def test_crop_augmentation_is_repeatable_for_epoch_and_index(fixture_root):
    a = DSECFlowDataset(fixture_root, ['synthetic_train'], crop_size=[20, 20], augment=True, seed=7)
    b = DSECFlowDataset(fixture_root, ['synthetic_train'], crop_size=[20, 20], augment=True, seed=7)
    for epoch in (0, 3):
        a.set_epoch(epoch)
        b.set_epoch(epoch)
        for key in ('old', 'new', 'flow', 'valid'):
            torch.testing.assert_close(a[1][key], b[1][key], rtol=0, atol=0)


@pytest.mark.parametrize('train,val', [(['a'], ['a']), ([], ['b']), (['a', 'a'], ['b']), ('abc', ['d'])])
def test_sequence_leakage_and_malformed_splits_rejected(train, val):
    with pytest.raises(ValueError):
        validate_split(train, val)


def test_sequence_split_accepts_disjoint_data():
    validate_split(['a', 'b'], ['c'])


def test_window_mismatch_and_crop_shape_errors(fixture_root):
    with pytest.raises(ValueError, match='differs'):
        DSECFlowDataset(fixture_root, ['synthetic_train'], window_s=.005)
    with pytest.raises(ValueError, match='crop_size'):
        DSECFlowDataset(fixture_root, ['synthetic_train'], crop_size=[12.5, 8])
    with pytest.raises(ValueError, match='single directory'):
        DSECFlowDataset(fixture_root, ['..'])
