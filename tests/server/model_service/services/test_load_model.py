from server.model_service.services.load_model import load_model

def test_load_model():
    model,tokenizer, flag = load_model()
    assert flag == True