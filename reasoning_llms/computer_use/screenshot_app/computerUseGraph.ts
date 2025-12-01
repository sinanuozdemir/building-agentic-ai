import { StateGraph, START, END, StateGraphArgs } from '@langchain/langgraph';
import { ChatOpenAI } from '@langchain/openai';
import { z } from 'zod';
import { app as electronApp, nativeImage, screen, desktopCapturer } from 'electron';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

// Define the state schema
const ComputerUseState = z.object({
  command: z.string(),
  screenshot: z.string(), // Base64 encoded screenshot
  reasoning_response: z.string().optional(),
  img_dimensions: z.string().optional(),
  action_result: z.string().optional(),
  tool_calls: z.array(z.any()).default([]),
  model: z.string().default(''),
});

type ComputerUseStateType = z.infer<typeof ComputerUseState>;

// State channels configuration
const graphStateChannels: StateGraphArgs<ComputerUseStateType>['channels'] = {
  command: {
    value: (prev: string, next: string) => next,
    default: () => '',
  },
  screenshot: {
    value: (prev: string, next: string) => next,
    default: () => '',
  },
  reasoning_response: {
    value: (prev: string | undefined, next: string | undefined) => next || prev,
  },
  img_dimensions: {
    value: (prev: string | undefined, next: string | undefined) => next || prev,
  },
  action_result: {
    value: (prev: string | undefined, next: string | undefined) => next || prev,
  },
  tool_calls: {
    value: (prev: any[], next: any[]) => next,
    default: () => [],
  },
  model: {
    value: (prev: string, next: string) => next,
    default: () => '',
  },
};


// PyAutoGUI-style tools for the LLM
const computerUseTools = [
  {
    type: 'function',
    function: {
      name: 'click',
      description: 'Click at specific coordinates on the screen',
      parameters: {
        type: 'object',
        properties: {
          x: { type: 'integer', description: 'X coordinate to click' },
          y: { type: 'integer', description: 'Y coordinate to click' },
        },
        required: ['x', 'y'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'double_click',
      description: 'Double click at specific coordinates on the screen',
      parameters: {
        type: 'object',
        properties: {
          x: { type: 'integer', description: 'X coordinate to double click' },
          y: { type: 'integer', description: 'Y coordinate to double click' },
        },
        required: ['x', 'y'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'type_text',
      description: 'Type text at the current cursor position',
      parameters: {
        type: 'object',
        properties: {
          text: { type: 'string', description: 'Text to type' },
        },
        required: ['text'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'press_key',
      description: 'Press a specific key',
      parameters: {
        type: 'object',
        properties: {
          key: { type: 'string', description: 'Key to press (e.g., "enter", "tab", "escape")' },
        },
        required: ['key'],
      },
    },
  },
];

// Node 1: Take Screenshot
async function takeScreenshotNode(state: ComputerUseStateType): Promise<Partial<ComputerUseStateType>> {
  console.log(`📸 Taking screenshot for command: ${state.command}`);
  
  try {
    // Get screen size
    const primaryDisplay = screen.getPrimaryDisplay();
    const { width, height } = primaryDisplay.bounds;
    console.log(`Screen size: ${width}x${height}`);
    
    // Take screenshot using Electron's desktopCapturer
    const sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width, height }
    });
    
    if (sources.length === 0) {
      throw new Error('No screen sources available');
    }
    
    // Get the first (primary) screen
    const screenshot = sources[0].thumbnail;
    const base64Screenshot = screenshot.toPNG().toString('base64');
    
    return {
      screenshot: base64Screenshot,
      img_dimensions: `The image is ${width}x${height}`,
    };
  } catch (error) {
    console.error('Error taking screenshot:', error);
    return {
      action_result: `Error taking screenshot: ${error}`,
    };
  }
}

// Node 2: Reasoning with LLM
async function reasoningNode(state: ComputerUseStateType): Promise<Partial<ComputerUseStateType>> {
  console.log(`🧠 Analyzing screenshot and planning action for: ${state.command}`);
  // Initialize LLM client
  const client = new ChatOpenAI({
    apiKey: process.env.OPENROUTER_API_KEY,
    model: state.model,
    temperature: 0,
    configuration: {
      baseURL: 'https://openrouter.ai/api/v1',
    },
    reasoning: {
      effort: 'medium',
    },
  });

  try {
    const response = await client.invoke([
      {
        role: 'user',
        content: [
          {
            type: 'text',
            text: `You are a GUI automation assistant performing the action: "${state.command}"

Here is a screenshot of the current screen. ${state.img_dimensions}
Use the available tools to perform the appropriate GUI action. Be precise with coordinates and choose the most appropriate action type.

First, reason through the screenshot provided and which tools you will need to accomplish the action.
Second, select a tool to use.`,
          },
          {
            type: 'image_url',
            image_url: {
              url: `data:image/png;base64,${state.screenshot}`,
            },
          },
        ],
      },
    ], {
      tools: computerUseTools,
      reasoning: {
        effort: 'medium',
      }
      
    });
    console.log('Response:', response);
    return {
      reasoning_response: response.content as string,
      tool_calls: response.tool_calls || [],
    };
  } catch (error) {
    console.error('Error in reasoning:', error);
    return {
      reasoning_response: `Error: ${error}`,
      tool_calls: [],
    };
  }
}

// Node 3: Execute Action
async function executeActionNode(state: ComputerUseStateType): Promise<Partial<ComputerUseStateType>> {
  console.log('⚡ Executing GUI action...');
  
  if (!state.tool_calls || state.tool_calls.length === 0) {
    return {
      action_result: `No action determined by AI. Reasoning: ${state.reasoning_response}`,
    };
  }
  
  try {
    const results: string[] = [];
    
    for (const toolCall of state.tool_calls) {
      console.log('Tool call:', toolCall);
      const functionName = toolCall.name;
      const toolCallId = toolCall.id;
      const args = toolCall.args;
      
      console.log(`Executing ${functionName} with args:`, args);
      
      switch (functionName) {
        case 'click':
          await execAsync(`osascript -e 'tell application "System Events" to click at {${args.x}, ${args.y}}'`);
          results.push(`Clicked at (${args.x}, ${args.y})`);
          break;
          
        case 'double_click':
          await execAsync(`osascript -e 'tell application "System Events" to double click at {${args.x}, ${args.y}}'`);
          results.push(`Double-clicked at (${args.x}, ${args.y})`);
          break;
          
        case 'type_text':
          const escapedText = args.text.replace(/"/g, '\\"');
          await execAsync(`osascript -e 'tell application "System Events" to keystroke "${escapedText}"'`);
          results.push(`Typed: '${args.text}'`);
          break;
          
        case 'press_key':
          const keyName = args.key === 'enter' ? 'return' : args.key;
          await execAsync(`osascript -e 'tell application "System Events" to key code "${keyName}"'`);
          results.push(`Pressed key: ${args.key}`);
          break;
          
        default:
          results.push(`Unknown function: ${functionName}`);
      }
      
      // Small delay between actions
      await new Promise(resolve => setTimeout(resolve, 500));
    }
    
    return {
      action_result: results.join('; '),
    };
  } catch (error) {
    const errorMsg = `Error executing action: ${error}`;
    console.error(errorMsg);
    return {
      action_result: errorMsg,
    };
  }
}

// Create the LangGraph workflow
function createComputerUseGraph() {
  const workflow = new StateGraph({
    channels: graphStateChannels,
  })
    .addNode('take_screenshot', takeScreenshotNode)
    .addNode('reasoning', reasoningNode)
    .addNode('execute_action', executeActionNode)
    .addEdge(START, 'take_screenshot')
    .addEdge('take_screenshot', 'reasoning')
    .addEdge('reasoning', 'execute_action')
    .addEdge('execute_action', END);
  
  return workflow.compile();
}

// Main function to execute GUI automation
export async function executeComputerUseCommand(command: string): Promise<any> {
  console.log(`🚀 Starting GUI automation for: '${command}'`);
  console.log('='.repeat(50));
  
  const automationGraph = createComputerUseGraph();
  
  const initialState = {
    command,
    screenshot: '',
    model: 'anthropic/claude-opus-4',
    tool_calls: [],
  };
  console.log('Initial state:', initialState);
  const result = await automationGraph.invoke(initialState);
  
  console.log('='.repeat(50));
  console.log('✅ Automation completed!');
  console.log(`Command: ${result.command}`);
  console.log(`🧠 AI Reasoning: ${result.reasoning_response}`);
  console.log(`⚡ Action Result: ${result.action_result}`);
  
  return result;
}

// Generic function to execute any automation command
export async function runAutomationCommand(command: string): Promise<any> {
  return executeComputerUseCommand(command);
}
