import {
  AutoModelForCausalLM,
  AutoTokenizer,
  TextStreamer,
  env,
} from "@huggingface/transformers";

// VL model support (may fail if onnxruntime-web has issues)
let VLModel = null;
try {
  const vlModule = await import("./vl-model.js");
  VLModel = vlModule.VLModel;
} catch (e) {
  console.warn("VL model support not available:", e);
}

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
const imageBtn = document.getElementById('imageBtn');
const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const dropOverlay = document.getElementById('dropOverlay');
const modelTypeBadge = document.getElementById('modelTypeBadge');

// State
let model = null;
let tokenizer = null;
let vlModel = null;  // Custom VL model instance
let pastKeyValues = null;
let messages = [];
let isGenerating = false;
let pendingImages = []; // Images to send with next message
let isVLModel = false;

// Model configurations
const MODELS = {
  // Text models
  'LFM2-350M-Q4': {
    path: '/models/LFM2-350M-ONNX-builder-Q4-fp32head',
    label: 'LFM2-350M Q4 (459 MB)',
    type: 'text',
  },
  'LFM2-700M-Q4': {
    path: '/models/LFM2-700M-ONNX-builder-Q4-fp32head',
    label: 'LFM2-700M Q4 (803 MB)',
    type: 'text',
  },
  'LFM2-1.2B-Q4': {
    path: '/models/LFM2-1.2B-ONNX-builder-Q4-fp32head',
    label: 'LFM2-1.2B Q4 (1.1 GB)',
    type: 'text',
  },
  'LFM2-2.6B-Q4': {
    path: '/models/LFM2-2.6B-ONNX-builder-Q4-fp32head',
    label: 'LFM2-2.6B Q4 (2.0 GB)',
    type: 'text',
  },
  // Vision-Language models
  'LFM2-VL-450M-B4V4': {
    path: '/models/LFM2-VL-450M-ONNX-B4V4',
    label: 'LFM2-VL-450M B4V4 (470 MB)',
    type: 'vl',
  },
  'LFM2-VL-1.6B-B4V4': {
    path: '/models/LFM2-VL-1.6B-ONNX-B4V4',
    label: 'LFM2-VL-1.6B B4V4 (1.2 GB)',
    type: 'vl',
  },
  'LFM2-VL-3B-B4V4': {
    path: '/models/LFM2-VL-3B-ONNX-B4V4',
    label: 'LFM2-VL-3B B4V4 (2.2 GB)',
    type: 'vl',
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
  imageBtn.disabled = !ready || !isVLModel;
}

function showProgress(show) {
  progressBar.style.display = show ? 'block' : 'none';
}

function updateProgress(percent, text) {
  progressFill.style.width = `${percent}%`;
  progressText.textContent = text || `${percent}%`;
}

function updateModelTypeBadge() {
  const modelKey = modelSelect.value;
  const config = MODELS[modelKey];
  if (config?.type === 'vl') {
    modelTypeBadge.innerHTML = '<span class="model-type-badge vl">Vision</span>';
  } else {
    modelTypeBadge.innerHTML = '<span class="model-type-badge text">Text</span>';
  }
}

function addMessage(role, content, images = [], isStreaming = false) {
  const msgEl = document.createElement('div');
  msgEl.className = `message ${role}${isStreaming ? ' generating' : ''}`;

  // Add images first
  for (const img of images) {
    const imgEl = document.createElement('img');
    imgEl.src = img.dataUrl;
    msgEl.appendChild(imgEl);
  }

  // Add text
  const textEl = document.createElement('span');
  textEl.textContent = content;
  msgEl.appendChild(textEl);

  chatContainer.appendChild(msgEl);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return { msgEl, textEl };
}

function clearPendingImages() {
  pendingImages = [];
  imagePreview.innerHTML = '';
}

function addPendingImage(dataUrl) {
  const id = Date.now() + Math.random();
  pendingImages.push({ id, dataUrl });

  const item = document.createElement('div');
  item.className = 'image-preview-item';
  item.dataset.id = id;

  const img = document.createElement('img');
  img.src = dataUrl;
  item.appendChild(img);

  const removeBtn = document.createElement('button');
  removeBtn.className = 'remove-btn';
  removeBtn.textContent = '×';
  removeBtn.onclick = () => {
    pendingImages = pendingImages.filter(i => i.id !== id);
    item.remove();
  };
  item.appendChild(removeBtn);

  imagePreview.appendChild(item);
}

async function loadImageFile(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => resolve(e.target.result);
    reader.readAsDataURL(file);
  });
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

    isVLModel = modelConfig.type === 'vl';

    if (isVLModel) {
      // Load VL model using custom ONNX Runtime implementation
      if (!VLModel) {
        throw new Error('VL model support not available. Check console for errors.');
      }
      setStatus('Loading vision-language model...');
      vlModel = new VLModel();
      await vlModel.load(modelConfig.path, {
        device: 'webgpu',
        progressCallback: (progress) => {
          if (progress.status === 'loading') {
            updateProgress(progress.progress, `Loading ${progress.file}...`);
          } else if (progress.status === 'done') {
            updateProgress(100, 'Done');
          }
        },
      });
      tokenizer = vlModel.tokenizer;
      model = vlModel;  // For compatibility
    } else {
      // Load text model
      setStatus('Loading tokenizer...');
      tokenizer = await AutoTokenizer.from_pretrained(modelConfig.path, {
        progress_callback: progressCallback,
      });

      setStatus('Loading model...');
      model = await AutoModelForCausalLM.from_pretrained(modelConfig.path, {
        device: 'webgpu',
        progress_callback: progressCallback,
      });
    }

    showProgress(false);
    const typeLabel = isVLModel ? 'Vision-Language' : 'Text';
    setStatus(`Ready! ${typeLabel} model loaded on WebGPU`, 'success');
    setReady(true);
    messages = [];
    pastKeyValues = null;
    clearPendingImages();

  } catch (error) {
    console.error('Load error:', error);
    showProgress(false);
    const msg = error instanceof Error ? error.message : String(error);
    setStatus(`Error: ${msg}`, 'error');
    model = null;
    tokenizer = null;
    vlModel = null;
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

  const imagesToSend = [...pendingImages];
  clearPendingImages();

  // Add user message to history (simple string content for both VL and text)
  messages.push({ role: 'user', content: userMessage });
  addMessage('user', userMessage, imagesToSend);

  const { msgEl, textEl } = addMessage('assistant', '', [], true);
  let generatedText = '';
  const startTime = performance.now();
  let tokenCount = 0;

  try {
    if (isVLModel) {
      // Use custom VL model with ONNX Runtime
      const imageUrls = imagesToSend.map(img => img.dataUrl);

      generatedText = await vlModel.generate(messages, {
        maxNewTokens: 512,
        images: imageUrls,
        onToken: (token, tokenId) => {
          if (token.includes('<|im_end|>') || token.includes('<|endoftext|>')) {
            return true;  // Stop
          }
          generatedText += token;
          tokenCount++;
          textEl.textContent = generatedText;
          chatContainer.scrollTop = chatContainer.scrollHeight;
          return false;
        },
      });

    } else {
      // Text model using transformers.js
      const input = tokenizer.apply_chat_template(messages, {
        add_generation_prompt: true,
        return_dict: true,
      });

      // Text streamer for real-time output
      const streamer = new TextStreamer(tokenizer, {
        skip_prompt: true,
        skip_special_tokens: false,
        callback_function: (token) => {
          if (token.includes('<|im_end|>') || token.includes('<|endoftext|>')) {
            return;
          }
          generatedText += token;
          tokenCount++;
          textEl.textContent = generatedText;
          chatContainer.scrollTop = chatContainer.scrollHeight;
        },
      });

      // Generate response
      const generateOptions = {
        ...input,
        max_new_tokens: 512,
        do_sample: false,
        streamer,
        return_dict_in_generate: true,
        past_key_values: pastKeyValues,
      };

      const result = await model.generate(generateOptions);

      // Update past key values for text models
      if (result.past_key_values) {
        pastKeyValues = result.past_key_values;
      }
    }

    // Clean up the generated text
    generatedText = generatedText.replace(/<\|im_end\|>$/g, '').trim();

    const elapsed = (performance.now() - startTime) / 1000;
    const tokensPerSec = tokenCount / elapsed;

    msgEl.classList.remove('generating');
    textEl.textContent = generatedText;

    const statsEl = document.createElement('div');
    statsEl.className = 'stats';
    statsEl.textContent = `${tokenCount} tokens in ${elapsed.toFixed(1)}s (${tokensPerSec.toFixed(1)} tok/s)`;
    msgEl.appendChild(statsEl);

    messages.push({ role: 'assistant', content: generatedText });

  } catch (error) {
    console.error('Generation error:', error);
    textEl.textContent = `Error: ${error.message}`;
    msgEl.classList.remove('generating');
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
modelSelect.addEventListener('change', updateModelTypeBadge);

clearBtn.addEventListener('click', () => {
  messages = [];
  pastKeyValues = null;
  chatContainer.innerHTML = '';
  clearPendingImages();
});

sendBtn.addEventListener('click', () => {
  const text = userInput.value.trim();
  if (text || pendingImages.length > 0) {
    userInput.value = '';
    generate(text || 'What do you see in this image?');
  }
});

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendBtn.click();
  }
});

// Image handling
imageBtn.addEventListener('click', () => {
  imageInput.click();
});

imageInput.addEventListener('change', async (e) => {
  for (const file of e.target.files) {
    if (file.type.startsWith('image/')) {
      const dataUrl = await loadImageFile(file);
      addPendingImage(dataUrl);
    }
  }
  imageInput.value = '';
});

// Paste handler
document.addEventListener('paste', async (e) => {
  if (!isVLModel || isGenerating) return;

  const items = e.clipboardData?.items;
  if (!items) return;

  for (const item of items) {
    if (item.type.startsWith('image/')) {
      e.preventDefault();
      const file = item.getAsFile();
      if (file) {
        const dataUrl = await loadImageFile(file);
        addPendingImage(dataUrl);
      }
    }
  }
});

// Drag and drop handlers
document.addEventListener('dragenter', (e) => {
  if (!isVLModel || isGenerating) return;
  e.preventDefault();
  dropOverlay.classList.add('active');
});

dropOverlay.addEventListener('dragleave', (e) => {
  e.preventDefault();
  dropOverlay.classList.remove('active');
});

dropOverlay.addEventListener('dragover', (e) => {
  e.preventDefault();
});

dropOverlay.addEventListener('drop', async (e) => {
  e.preventDefault();
  dropOverlay.classList.remove('active');

  if (!isVLModel || isGenerating) return;

  const files = e.dataTransfer?.files;
  if (!files) return;

  for (const file of files) {
    if (file.type.startsWith('image/')) {
      const dataUrl = await loadImageFile(file);
      addPendingImage(dataUrl);
    }
  }
});

// Initialize
updateModelTypeBadge();

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
