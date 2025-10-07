# test_evaluate_test_set.py

import pytest
import sys
import types

from taiwan_food_cnn.evaluate_test_set import evaluate_on_test_set

@pytest.fixture
def mock_predictions():
    # Simulate two sets of predictions and confidences
    preds1 = [1, 2, 3, 4, 5]
    confs1 = [0.8, 0.6, 0.9, 0.4, 0.7]
    preds2 = [2, 2, 1, 4, 5]
    confs2 = [0.7, 0.9, 0.5, 0.8, 0.6]
    return (preds1, confs1), (preds2, confs2)

def select_highest_confidence(preds1, confs1, preds2, confs2):
    # Select prediction with highest confidence for each sample
    final_preds = []
    final_confs = []
    for p1, c1, p2, c2 in zip(preds1, confs1, preds2, confs2):
        if c1 >= c2:
            final_preds.append(p1)
            final_confs.append(c1)
        else:
            final_preds.append(p2)
            final_confs.append(c2)
    return final_preds, final_confs

def test_select_highest_confidence(mock_predictions):
    (preds1, confs1), (preds2, confs2) = mock_predictions
    final_preds, final_confs = select_highest_confidence(preds1, confs1, preds2, confs2)
    assert final_preds == [1, 2, 3, 4, 5]
    assert final_confs == [0.8, 0.9, 0.9, 0.8, 0.7]

def test_all_equal_confidence():
    preds1 = [1, 2, 3]
    confs1 = [0.5, 0.5, 0.5]
    preds2 = [3, 2, 1]
    confs2 = [0.5, 0.5, 0.5]
    final_preds, final_confs = select_highest_confidence(preds1, confs1, preds2, confs2)
    assert final_preds == preds1
    assert final_confs == confs1

def test_missing_predictions():
    preds1 = [1, None, 3]
    confs1 = [0.8, 0.6, 0.9]
    preds2 = [2, 2, None]
    confs2 = [0.7, 0.9, 0.5]
    final_preds, final_confs = select_highest_confidence(preds1, confs1, preds2, confs2)
    assert final_preds == [1, 2, 3]
    assert final_confs == [0.8, 0.9, 0.9]

def test_evaluate_on_test_set(monkeypatch, tmp_path, mock_predictions):
    # Patch file existence and model loading
    monkeypatch.setattr('os.path.exists', lambda x: True)
    monkeypatch.setattr('os.listdir', lambda x: ['model1.pth', 'model2.pth'])
    monkeypatch.setattr('os.path.getctime', lambda x: 1 if 'model1' in x else 2)
    # Patch get_model and torch.load
    sys.modules['pytorch_model'] = types.SimpleNamespace(get_model=lambda *a, **kw: types.SimpleNamespace(
        to=lambda x: x, eval=lambda: None, load_state_dict=lambda x: None
    ))
    sys.modules['pytorch_data_loader'] = types.SimpleNamespace(TaiwanFoodDataset=lambda *a, **kw: [0, 1, 2, 3, 4])
    # Patch DataLoader and transforms
    sys.modules['torch.utils.data'] = types.SimpleNamespace(DataLoader=lambda *a, **kw: [[0, 1, 2, 3, 4]])
    sys.modules['torchvision'] = types.SimpleNamespace(transforms=types.SimpleNamespace(Compose=lambda x: x))
    # Patch tqdm
    sys.modules['tqdm'] = types.SimpleNamespace(tqdm=lambda x, **kw: x)
    # Patch torch.device
    sys.modules['torch'] = types.SimpleNamespace(device=lambda x: x, no_grad=lambda: types.SimpleNamespace(__enter__=lambda x: None, __exit__=lambda x, y, z, w: None))
    # Patch prediction logic to use mock_predictions
    preds1, confs1 = mock_predictions[0]
    preds2, confs2 = mock_predictions[1]
    def fake_evaluate_with_amd_npu(*args, **kwargs):
        return select_highest_confidence(preds1, confs1, preds2, confs2)
    def fake_evaluate_standard_mode(*args, **kwargs):
        return preds1, confs1
    monkeypatch.setattr('taiwan_food_cnn.evaluate_test_set.evaluate_with_amd_npu', fake_evaluate_with_amd_npu)
    monkeypatch.setattr('taiwan_food_cnn.evaluate_test_set.evaluate_standard_mode', fake_evaluate_standard_mode)
    # Run
    result = evaluate_on_test_set(
        model_path='models/model2.pth',
        test_csv='train_list.csv',
        test_img_dir='train',
        num_classes=5,
        batch_size=2,
        img_size=224,
        manual_device_selection=False
    )
    assert result == ([1, 2, 3, 4, 5], [0.8, 0.9, 0.9, 0.8, 0.7])