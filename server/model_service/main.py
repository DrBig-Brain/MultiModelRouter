from fastapi import FastAPI
from services.load_model import load_model
from services.inference import inference
from fastapi.responses import JSONResponse

app = FastAPI()

model,tokenizer,_ = load_model()


@app.get("/")
def root():
    return {
        "name" : "Model-service",
        "status" : "running"
    }

@app.get("/infer")
def infer(
    prompt : str
):
    try:

        results = inference(model,
            tokenizer,
            prompt
        )

        return {
            "status" : 200,
            "prompt_scores" : results
        }
    except Exception as e:
        return {
            "status" : 500,
            "message" : e
        }



if __name__ == "__main__":


    import uvicorn
    uvicorn.run(
        "main:app",
        host = "0.0.0.0",
        port = 8001,
        log_level = "info"
    )
