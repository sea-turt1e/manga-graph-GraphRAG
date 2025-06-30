# Manga GraphRAG API Sample

This sample demonstrates how to implement GraphRAG (Graph Retrieval-Augmented Generation) using the Manga Graph API.

## Features

- **Manga Recommendations**: Get personalized manga recommendations based on user preferences
- **Manga Analysis**: Deep analysis of specific manga titles using graph data
- **Multi-hop Relationships**: Discover complex relationships between entities in the manga knowledge graph
- **Author Lineage**: Explore author influences and lineage in the manga industry

## Prerequisites

1. **Manga Graph API**: The API should be running at `http://localhost:8000`
2. **OpenAI API Key**: Set your OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```
3. **Python Dependencies**: Install required packages:
   ```bash
   pip install requests langchain langchain-openai python-dotenv
   ```

## Usage

### Basic Demo

Run the default demo to see all GraphRAG capabilities:

```bash
python sample_with_manga-graph-api.py
```

### Interactive Mode

For an interactive experience where you can explore different features:

```bash
python sample_with_manga-graph-api.py --interactive
```

Available commands in interactive mode:
- `recommend <query>` - Get manga recommendations
- `analyze <title>` - Analyze a specific manga
- `relate <entity1> | <entity2>` - Find relationships between entities
- `lineage <author>` - Explore author lineage
- `help` - Show available commands
- `quit` - Exit the demo

## Example Queries

### Recommendation
```
recommend ONE PIECEが好きです。似たような冒険漫画を教えてください。
recommend NARUTOが好きです。忍者や友情をテーマにした作品を探しています。
recommend るろうに剣心が好きです。歴史物や剣術をテーマにした作品を教えてください。
```

### Analysis
```
analyze ONE PIECE
analyze NARUTO
analyze るろうに剣心
```

### Relationships
```
relate 尾田栄一郎 | ONE PIECE
relate 岸本斉史 | NARUTO
relate 和月伸宏 | るろうに剣心
```

### Author Lineage
```
lineage 尾田栄一郎
lineage 岸本斉史
lineage 和月伸宏
```

## Architecture

The implementation consists of:

1. **MangaGraphClient**: Handles communication with the Manga Graph API
2. **MangaGraphRAG**: Implements GraphRAG logic using LangChain and OpenAI
3. **Prompt Templates**: Customized prompts for different GraphRAG tasks

## API Endpoints Used

- `/api/v1/neo4j/search` - Search the Neo4j graph database
- `/api/v1/neo4j/creator/{name}` - Get works by a specific creator
- `/api/v1/media-arts/search-with-related` - Search with related works
- `/api/v1/neo4j/stats` - Get database statistics

## Error Handling

If you encounter errors:
1. Ensure the Manga Graph API is running at `http://localhost:8000`
2. Verify your OpenAI API key is correctly set in `.env`
3. Check that all required Python packages are installed