from server.model_service.services.inference import inference
from server.model_service.services.load_model import load_model
import numpy as np
def test_inference():
    model, tokenizer, flag = load_model()
    assert inference(model, tokenizer, "Who is the president of India") == {'task_type_1': ['Open QA'], 'task_type_2': ['NA'], 'task_type_prob': [0.981], 'creativity_scope': [np.float64(0.0117)], 'reasoning': [np.float64(0.0053)], 'contextual_knowledge': [np.float64(0.0597)], 'number_of_few_shots': [0], 'domain_knowledge': [np.float64(0.3666)], 'no_label_reason': [np.float64(0.0)], 'constraint_ct': [np.float64(0.0109)], 'prompt_complexity_score': [np.float64(0.06503)]}