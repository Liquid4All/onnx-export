import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
} from "@huggingface/transformers";

// Enable local model loading
env.allowLocalModels = true;
env.useBrowserCache = false;

// DOM elements
const modelSelect = document.getElementById('modelSelect');
const loadBtn = document.getElementById('loadBtn');
const clearBtn = document.getElementById('clearBtn');
const statusEl = document.getElementById('status');
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const sendBtn = document.getElementById('sendBtn');
const progressBar = document.getElementById('progressBar');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');

// State
let model = null;
let tokenizer = null;
let pastKeyValues = null;
let messages = [];
let isGenerating = false;

// Model configurations - served via /models/ alias
// Note: WebGPU only supports 4-bit MatMulNBits, Q8 not supported
const MODELS = {
  'LFM2-350M-Q4': {
    path: '/models/LFM2-350M-ONNX-builder-Q4-fp32head',
    label: 'LFM2-350M Q4 (459 MB)',
  },
  'LFM2-700M-Q4': {
    path: '/models/LFM2-700M-ONNX-builder-Q4-fp32head',
    label: 'LFM2-700M Q4 (803 MB)',
  },
  'LFM2-1.2B-Q4': {
    path: '/models/LFM2-1.2B-ONNX-builder-Q4-fp32head',
    label: 'LFM2-1.2B Q4 (1.1 GB)',
  },
  'LFM2-2.6B-Q4': {
    path: '/models/LFM2-2.6B-ONNX-builder-Q4-fp32head',
    label: 'LFM2-2.6B Q4 (2.0 GB)',
  },
};

// ============================================================================
// UI Helpers
// ============================================================================
function setStatus(text, type = '') {
  statusEl.textContent = text;
  statusEl.className = type;
}

function setLoading(loading) {
  loadBtn.disabled = loading;
  modelSelect.disabled = loading;
}

function setReady(ready) {
  userInput.disabled = !ready;
  sendBtn.disabled = !ready;
}

function showProgress(show) {
  progressBar.style.display = show ? 'block' : 'none';
}

function updateProgress(percent, text) {
  progressFill.style.width = `${percent}%`;
  progressText.textContent = text || `${percent}%`;
}

function addMessage(role, content, isStreaming = false) {
  const msgEl = document.createElement('div');
  msgEl.className = `message ${role}${isStreaming ? ' generating' : ''}`;
  msgEl.textContent = content;
  chatContainer.appendChild(msgEl);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return msgEl;
}

// ============================================================================
// Model Loading
// ============================================================================
async function loadModel() {
  const modelKey = modelSelect.value;
  const modelConfig = MODELS[modelKey];

  if (!modelConfig) {
    setStatus('Invalid model selection', 'error');
    return;
  }

  setLoading(true);
  setReady(false);
  showProgress(true);
  updateProgress(0, 'Starting...');
  setStatus(`Loading ${modelConfig.label}...`);

  try {
    // Verify WebGPU
    if (!navigator.gpu) {
      throw new Error('WebGPU not available. Enable at chrome://flags/#enable-unsafe-webgpu');
    }

    // Progress callback
    const progressCallback = (progress) => {
      if (progress.status === 'progress') {
        const percent = Math.round((progress.loaded / progress.total) * 100);
        const fileName = progress.file?.split('/').pop() || '';
        updateProgress(percent, `${fileName}: ${percent}%`);
      } else if (progress.status === 'done') {
        updateProgress(100, 'Done');
      }
    };

    // Load tokenizer
    setStatus('Loading tokenizer...');
    tokenizer = await AutoTokenizer.from_pretrained(modelConfig.path, {
      progress_callback: progressCallback,
    });

    // Load model
    setStatus('Loading model (this may take a while)...');
    model = await AutoModelForCausalLM.from_pretrained(modelConfig.path, {
      device: 'webgpu',
      progress_callback: progressCallback,
    });

    showProgress(false);
    setStatus(`Ready! Model loaded on WebGPU`, 'success');
    setReady(true);
    messages = [];
    pastKeyValues = null;

  } catch (error) {
    console.error('Load error:', error);
    showProgress(false);
    const msg = error instanceof Error ? error.message : String(error);
    setStatus(`Error: ${msg}`, 'error');
    model = null;
    tokenizer = null;
  } finally {
    setLoading(false);
  }
}

// ============================================================================
// Generation
// ============================================================================
async function generate(userMessage) {
  if (!model || !tokenizer || isGenerating) return;

  isGenerating = true;
  setReady(false);

  messages.push({ role: 'user', content: userMessage });
  addMessage('user', userMessage);

  const assistantMsgEl = addMessage('assistant', '', true);
  let generatedText = '';
  const startTime = performance.now();
  let tokenCount = 0;

  try {
    // Apply chat template
    const input = tokenizer.apply_chat_template(messages, {
      add_generation_prompt: true,
      return_dict: true,
    });

    // Text streamer for real-time output
    const streamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: false,
      callback_function: (token) => {
        // Skip end tokens
        if (token.includes('<|im_end|>') || token.includes('<|endoftext|>')) {
          return;
        }
        generatedText += token;
        tokenCount++;
        assistantMsgEl.textContent = generatedText;
        chatContainer.scrollTop = chatContainer.scrollHeight;
      },
    });

    // Generate response
    const { sequences, past_key_values } = await model.generate({
      ...input,
      past_key_values: pastKeyValues,
      max_new_tokens: 512,
      do_sample: false,
      streamer,
      return_dict_in_generate: true,
    });

    // Update past key values for next turn
    pastKeyValues = past_key_values;

    // Clean up the generated text
    generatedText = generatedText.replace(/<\|im_end\|>$/g, '').trim();

    const elapsed = (performance.now() - startTime) / 1000;
    const tokensPerSec = tokenCount / elapsed;

    assistantMsgEl.classList.remove('generating');
    assistantMsgEl.textContent = generatedText;

    const statsEl = document.createElement('div');
    statsEl.className = 'stats';
    statsEl.textContent = `${tokenCount} tokens in ${elapsed.toFixed(1)}s (${tokensPerSec.toFixed(1)} tok/s)`;
    assistantMsgEl.appendChild(statsEl);

    messages.push({ role: 'assistant', content: generatedText });

  } catch (error) {
    console.error('Generation error:', error);
    assistantMsgEl.textContent = `Error: ${error.message}`;
    assistantMsgEl.classList.remove('generating');
    messages.pop();
  } finally {
    isGenerating = false;
    setReady(true);
    userInput.focus();
  }
}

// ============================================================================
// Event Handlers
// ============================================================================
loadBtn.addEventListener('click', loadModel);

clearBtn.addEventListener('click', () => {
  messages = [];
  pastKeyValues = null;
  chatContainer.innerHTML = '';
});

sendBtn.addEventListener('click', () => {
  const text = userInput.value.trim();
  if (text) {
    userInput.value = '';
    generate(text);
  }
});

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// Check WebGPU on load
(async () => {
  if (!navigator.gpu) {
    setStatus('WebGPU not available. Enable at chrome://flags/#enable-unsafe-webgpu and restart Chrome.', 'error');
    return;
  }

  try {
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) {
      setStatus('WebGPU adapter not found. Check chrome://gpu for WebGPU status.', 'error');
      return;
    }

    const info = adapter.info || {};
    const desc = info.description || info.vendor || info.architecture || 'Available';
    setStatus(`WebGPU: ${desc}. Select model and click Load.`);
  } catch (e) {
    setStatus(`WebGPU error: ${e.message}. Try chrome://flags/#enable-unsafe-webgpu`, 'error');
  }
})();
