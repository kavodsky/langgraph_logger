# FastAPI LangGraph Logger Example

This example demonstrates how to integrate LangGraph Logger with FastAPI for tracking graph executions in a web API.

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up your OpenAI API key:**
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

Or modify the `OPENAI_API_KEY` variable in `fastapi_example.py`.

3. **Run the server:**
```bash
python fastapi_example.py
```

The server will start on `http://localhost:8000`.

## API Endpoints

### 1. Execute Graph (Synchronous)
```bash
curl -X POST "http://localhost:8000/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "Hello, how are you?",
    "user_id": "user123",
    "session_id": "session456",
    "tags": ["test", "demo"],
    "extra_metadata": {
      "source": "api",
      "priority": "high"
    }
  }'
```

### 2. Execute Graph (Asynchronous)
```bash
curl -X POST "http://localhost:8000/execute-async" \
  -H "Content-Type: application/json" \
  -d '{
    "initial_message": "Process this in background",
    "user_id": "user123",
    "tags": ["background", "async"]
  }'
```

### 3. List Executions
```bash
# Get all executions
curl "http://localhost:8000/executions"

# With filters
curl "http://localhost:8000/executions?limit=5&status=completed&graph_name=chat_processing_graph"
```

### 4. Get Execution Details
```bash
curl "http://localhost:8000/executions/{execution_id}"
```

### 5. Get Recovery Information
```bash
curl "http://localhost:8000/executions/{execution_id}/recovery"
```

### 6. Get Statistics
```bash
curl "http://localhost:8000/stats?days=7"
```

## Example Responses

### Successful Execution
```json
{
  "execution_id": "123e4567-e89b-12d3-a456-426614174000",
  "status": "completed",
  "result": {
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?",
        "timestamp": "1703123456.789"
      },
      {
        "role": "assistant", 
        "content": "Hello! I'm doing well, thank you for asking...",
        "timestamp": "1703123457.123"
      }
    ],
    "current_step": "completed",
    "processed_count": 3
  }
}
```

### Execution List
```json
{
  "executions": [
    {
      "id": "123e4567-e89b-12d3-a456-426614174000",
      "graph_name": "chat_processing_graph",
      "status": "completed",
      "started_at": "2023-12-21T10:30:00",
      "completed_at": "2023-12-21T10:30:05",
      "duration_seconds": 5.234,
      "total_nodes": 3,
      "completed_nodes": 3,
      "failed_nodes": 0,
      "success_rate": 100.0,
      "tags": ["test", "demo"]
    }
  ],
  "total": 1
}
```

## Interactive API Documentation

Once the server is running, visit:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

These provide interactive documentation where you can test the API endpoints directly.

## Database

The example uses SQLite by default with the database file `fastapi_graph_executions.db`. You can change this by modifying the `DATABASE_URL` variable in the code.

For production, consider using PostgreSQL:
```python
DATABASE_URL = "postgresql://user:password@localhost/langgraph_logger"
```

## Graph Structure

The example creates a simple chat processing graph with three nodes:

1. **preprocess** - Adds metadata to the input message
2. **llm** - Processes the message with ChatGPT (or mock response)
3. **postprocess** - Adds final metadata to the response

Each execution is tracked with:
- Node-level timing and status
- State snapshots for recovery
- Detailed metrics and statistics
- Error handling and recovery information

## Customization

You can easily modify the example to:

1. **Change the graph structure** - Modify the `create_chat_graph()` function
2. **Add authentication** - Use FastAPI dependencies for auth
3. **Add rate limiting** - Use slowapi or similar middleware
4. **Add caching** - Cache frequently accessed execution data
5. **Add webhooks** - Notify external systems of execution events
6. **Custom metadata** - Add domain-specific metadata to executions

## Production Considerations

For production deployment:

1. **Environment Variables** - Use proper environment variable management
2. **Database Connection Pooling** - Configure appropriate pool sizes
3. **Logging** - Set up structured logging with proper log levels
4. **Monitoring** - Add health checks and metrics endpoints
5. **Error Handling** - Implement comprehensive error handling
6. **Security** - Add authentication, rate limiting, and input validation
7. **Scaling** - Consider using async database drivers for better performance

## Testing

You can test the API using:

```python
import httpx
import asyncio

async def test_execution():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/execute",
            json={
                "initial_message": "Test message",
                "user_id": "test_user",
                "tags": ["test"]
            }
        )
        print(response.json())

asyncio.run(test_execution())
```