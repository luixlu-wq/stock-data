# How to Resume This Project in a New Conversation

## Quick Resume Method

When starting a new conversation with Claude Code, simply say:

```
"I'm working on a stock prediction project in c:\Users\luixj\AI\stock-data\.
Please read PROJECT_CONTEXT.md to understand what was built.
I want to [describe what you want to do next]."
```

Claude will read the context file and understand the entire project.

## What to Share Based on Your Goal

### If You're Setting Up for the First Time

**Say:**
```
"I have a stock prediction project that was created.
I'm at the setup phase and need help with [specific step]."
```

**Share:**
- This file: `RESUME_INSTRUCTIONS.md`
- Context file: `PROJECT_CONTEXT.md`
- Any errors you encounter

### If You're Adding Features

**Say:**
```
"I want to add [feature] to my stock prediction project.
Context is in PROJECT_CONTEXT.md.
The relevant files are [list files]."
```

**Share:**
- `PROJECT_CONTEXT.md`
- Relevant source files (e.g., `src/models/lstm_model.py`)
- `config/config.yaml` if changing settings

### If You're Debugging

**Say:**
```
"I'm getting an error when running [command].
Project context is in PROJECT_CONTEXT.md."
```

**Share:**
- `PROJECT_CONTEXT.md`
- Error message (full traceback)
- Command you ran
- Relevant log files from `logs/`

### If You're Extending the Project

**Say:**
```
"I want to implement [new feature] in my stock prediction project.
Background is in PROJECT_CONTEXT.md.
I'm thinking of [your approach]."
```

**Share:**
- `PROJECT_CONTEXT.md`
- Files that will be modified
- Your implementation ideas

## Example Conversations

### Example 1: Adding Backtesting
```
User: "I have a stock prediction project (context in PROJECT_CONTEXT.md).
I want to add a backtesting framework that simulates trading strategies
using the predictions. Can you help implement this?"

Claude will:
1. Read PROJECT_CONTEXT.md
2. Understand the existing architecture
3. Create a backtesting module
4. Integrate it with the existing code
5. Update the main.py with new commands
```

### Example 2: Improving Model
```
User: "My stock prediction model (context in PROJECT_CONTEXT.md) has
50% accuracy on classification. I want to try adding attention mechanism
to the LSTM. Can you help?"

Claude will:
1. Read the context
2. Review current LSTM implementation
3. Implement attention-based LSTM
4. Update the model architecture
5. Explain the changes
```

### Example 3: Deployment
```
User: "I want to deploy my stock prediction models (context in
PROJECT_CONTEXT.md) as a REST API using FastAPI. Can you help create
the API endpoints?"

Claude will:
1. Understand the project structure
2. Create FastAPI endpoints
3. Add model loading and inference
4. Create deployment scripts
5. Update documentation
```

## Files to Keep Handy

Always have these available when resuming:

1. **PROJECT_CONTEXT.md** - Project overview and decisions made
2. **README.md** - Full documentation
3. **config/config.yaml** - Current configuration
4. **main.py** - Entry point to understand commands
5. **logs/** - Error logs if debugging

## Tips for Effective Resumption

### ✅ DO:
- Mention the project location: `c:\Users\luixj\AI\stock-data\`
- Reference PROJECT_CONTEXT.md in your first message
- Be specific about what you want to do
- Share error messages in full
- Mention which files are relevant

### ❌ DON'T:
- Assume Claude remembers previous conversations (it doesn't)
- Skip mentioning the context file
- Be vague ("make it better")
- Share only partial error messages

## Common Next Steps

Here are likely things you'll want to do next:

### 1. Initial Setup and Run
```
"I need help setting up the environment for my stock prediction project.
Context in PROJECT_CONTEXT.md. I'm on Windows and have:
- Python 3.x installed
- Docker installed
- [GPU or no GPU]

Walk me through the setup steps."
```

### 2. Customization
```
"I want to change the stock tickers to [your tickers] and add
[specific technical indicator]. Project context in PROJECT_CONTEXT.md."
```

### 3. Model Improvement
```
"The model accuracy is [X%]. I want to try [approach].
Context in PROJECT_CONTEXT.md. Current model is in src/models/lstm_model.py."
```

### 4. Add Visualization
```
"I want to create visualizations of predictions vs actual prices.
Project context in PROJECT_CONTEXT.md. I want to visualize [specific things]."
```

### 5. Production Deployment
```
"I want to deploy this model to make real-time predictions.
Context in PROJECT_CONTEXT.md. I'm thinking of using [approach]."
```

## Project State Tracking

Update this section after significant changes:

**Last Modified**: 2026-01-01
**Current State**: Initial project creation complete, not yet run
**Next Step**: User setup and first run

**Recent Changes**:
- Created all project files
- Configured for 20 US stocks
- Set up temporal split (train on ≤2024, test on 2025)
- Implemented dual models (regression + classification)

**Known Issues**: None yet

**Pending Tasks**:
- [ ] User needs to setup environment
- [ ] User needs to get Polygon.io API key
- [ ] User needs to run first training

---

## Quick Command Reference

For quick reference when resuming:

```bash
# Check GPU
python main.py check-gpu

# Download data
python main.py download

# Preprocess
python main.py preprocess

# Upload to Qdrant
python main.py upload

# Train models
python main.py train-reg
python main.py train-clf

# Evaluate
python main.py eval-reg
python main.py eval-clf

# Run everything
python main.py all
```

---

**Remember**: Always mention `PROJECT_CONTEXT.md` in your first message when resuming!
