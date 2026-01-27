#!/bin/bash
# Current project (auto-detects)
# Usage: ./new-taskmaster-chat.sh [project] [branch]
# Example: ./new-taskmaster-chat.sh "RAG-Advanced" "dev"

# ./new-taskmaster-chat.sh

# # Specific project/branch
# ./new-taskmaster-chat.sh "RAG-Advanced" "dev"
# ./new-taskmaster-chat.sh "FinanceBot" "main"

# # Add to shell aliases  
# alias newtm="./new-taskmaster-chat.sh"
# alias newtmrag='./new-taskmaster-chat.sh "RAG-Advanced" "dev"'


PROJECT=${1:-"MyProject"}  # Default project name
BRANCH=${2:-"main"}        # Default branch

# Capture Task Master status
STATUS=$(task-master list 2>/dev/null || echo "No tasks found. Run 'task-master parse-prd' first.")

# Generate template
TEMPLATE=$(cat << EOF
You are continuing Task Master execution in Cursor Agent mode for $PROJECT

CRITICAL CONTEXT FILES (READ THESE FIRST):
- tasks.json: Full task list, status, deps—ALWAYS call \`task-master next\` for next task
- .cursor/rules/dev_workflow.mdc: Task Master rules
- .taskmaster/docs/prd.txt: PRD overview
- Main branch: $BRANCH

WORKFLOW:
1. Query \`task-master next\` for dependency-resolved task
2. Implement ONLY that task (YOLO mode: tests/builds allowed)
3. Test thoroughly
4. Call \`task-master complete --id=X\` when done
5. Repeat: "Next task"

Current status:
\`\`\`
$STATUS
\`\`\`

Start: "Show me the next task and implement it."
EOF
)

# Copy to clipboard
echo "$TEMPLATE" | pbcopy
echo "✅ Cursor Task Master template ready for $PROJECT ($BRANCH)"
echo "📋 Paste into new chat (Cmd+V). Status captured: $(echo "$STATUS" | head -3)"
