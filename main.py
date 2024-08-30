import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from functions import SearchApp

class UserRequestIn(BaseModel):
    query: str
    user_id: str

class SearchResult(BaseModel):
    output: str

app = FastAPI()
search_app = SearchApp()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search", response_model=SearchResult)
async def search(request_user: UserRequestIn):
    try:
        results = search_app.perform_vector_search(request_user.query)

        top_results = [
            {
                "chunk_id": result["chunk_id"],
                "title": result["title"],
                "chunk": result["chunk"]
            }
            for result in results
        ]

        llm_chain = search_app.get_llm_chain()
        output = llm_chain.run({"Question": request_user.query, "docs": top_results})
        output = str(output)

        search_app.log_search(request_user.user_id, request_user.query, output)

        return {"output": output}
    except Exception as e:
        print(f"Error in search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)