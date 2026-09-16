"""Dataset provenance from actual tiny NIfTI files; no real BraTS dependency."""
import csv
import json
import nibabel as nib
import numpy as np
import pytest
from evaluation.provenance import audit_dataset, sha256_file, split_identity, MODALITIES


@pytest.fixture
def data(tmp_path):
    root = tmp_path/'data'; root.mkdir()
    train, val = tmp_path/'train.txt', tmp_path/'val.txt'
    train.write_text('b\na\n'); val.write_text('c\n')
    for case in ('c','a','b'):
        directory = root/case; directory.mkdir()
        for modality in MODALITIES:
            array = np.resize(np.arange(5,dtype=np.int16),(2,3,4)) if modality == 'seg' else np.ones((2,3,4),np.float32)
            nib.save(nib.Nifti1Image(array,np.eye(4)),directory/f'{case}-{modality}.nii.gz')
    return root,train,val,tmp_path/'audit'


def audit(data,**kwargs):
    return audit_dataset(*data,expected_total=3,expected_train=2,expected_validation=1,**kwargs)


def test_observed_manifest_and_hashes(data):
    result = audit(data)
    assert result['audit_passed']
    assert result['train_validation_intersection_count'] == 0
    assert result['train_validation_union_count'] == 3
    assert result['observed_global_label_values'] == [0,1,2,3,4]
    assert result['per_label_case_presence'] == dict(NETC=3,SNFH=3,ET=3,RC=3)
    manifest = data[3]/'dataset_manifest.csv'
    assert result['manifest_sha256'] == sha256_file(manifest)
    rows = list(csv.DictReader(manifest.open()))
    assert [r['case_id'] for r in rows] == ['a','b','c']
    assert rows[0]['t1n_relative_path'] == 'a/a-t1n.nii.gz'
    assert audit(data)['manifest_sha256'] == result['manifest_sha256']
    hashed = audit(data,hash_dataset=True)
    rows = list(csv.DictReader(manifest.open()))
    assert rows[0]['seg_sha256'] == sha256_file(data[0]/'a/a-seg.nii.gz')
    assert hashed['manifest_sha256'] != result['manifest_sha256']


@pytest.mark.parametrize('issue', ['overlap','unknown','union','duplicate','count'])
def test_split_failures(data,issue):
    root,train,val,out = data
    if issue == 'overlap': val.write_text('a\n')
    if issue == 'unknown': val.write_text('unknown\n')
    if issue == 'union': train.write_text('a\n')
    if issue == 'duplicate': train.write_text('a\na\n')
    if issue == 'count': (root/'extra').mkdir()
    with pytest.raises(ValueError,match='Dataset audit failed'):
        audit(data)
    assert not json.loads((out/'dataset_audit.json').read_text())['audit_passed']


@pytest.mark.parametrize('modality',['t1n','seg'])
def test_missing_files(data,modality):
    (data[0]/'a'/f'a-{modality}.nii.gz').unlink()
    with pytest.raises(ValueError,match='missing'):
        audit(data)


@pytest.mark.parametrize('issue',['labels','shape','affine','spacing'])
def test_content_failures(data,issue):
    path = data[0]/'a/a-seg.nii.gz'
    array = np.full((2,3,5) if issue == 'shape' else (2,3,4),5 if issue == 'labels' else 0,np.int16)
    affine = np.eye(4)
    if issue == 'affine': affine[0,3] = 2
    if issue == 'spacing': affine[0,0] = 2
    nib.save(nib.Nifti1Image(array,affine),path)
    with pytest.raises(ValueError,match='Dataset audit failed'):
        audit(data)


def test_split_bytes_vs_normalized(data):
    path = data[1]
    _,before = split_identity(path)
    path.write_text('a\nb\n')
    _,after = split_identity(path)
    assert before['sha256'] != after['sha256']
    assert before['normalized_ids_sha256'] == after['normalized_ids_sha256']


def test_missing_split_and_invalid_geometry_metadata(data):
    data[1].unlink()
    with pytest.raises(ValueError,match='could not be read'):
        audit(data)
    result = json.loads((data[3]/'dataset_audit.json').read_text())
    assert result['train_split_sha256'] is None


def test_unexpected_files_reported_without_release_claim(data):
    (data[0]/'a'/'notes.txt').write_text('auxiliary file')
    result = audit(data)
    assert 'a/notes.txt' in result['unexpected_files']
    assert 'Declared' in result['release_authentication']
    assert 'NOT byte-level' in result['fingerprint_scope']


def test_segmentation_presence_is_observed(data):
    for folder in data[0].iterdir():
        nib.save(nib.Nifti1Image(np.full((2,3,4),4,np.int16),np.eye(4)),folder/f'{folder.name}-seg.nii.gz')
    result = audit(data)
    assert result['observed_global_label_values'] == [4]
    assert result['per_label_case_presence'] == dict(NETC=0,SNFH=0,ET=0,RC=3)
