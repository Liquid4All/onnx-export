/**
 * LFM2-VL Model Runner for ONNX Runtime Web
 *
 * Runs VL model inference using three ONNX models:
 * 1. embed_tokens.onnx - Text token embeddings
 * 2. embed_images.onnx - Image embeddings from patches
 * 3. decoder.onnx - Autoregressive decoder with conv state cache
 */

import * as ort from 'onnxruntime-web';
import { AutoTokenizer } from '@huggingface/transformers';
import { processImage, loadImage } from './vl-processor.js';

export class VLModel {
  constructor() {
    this.tokenizer = null;
    this.embedTokensSession = null;
    this.embedImagesSession = null;
    this.decoderSession = null;
    this.config = null;
    this.imageTokenId = null;
    this.eosTokenId = null;
    this.hiddenSize = 1024;  // Default for 450M
  }

  /**
   * Load the VL model from a directory
   * @param {string} modelPath - Path to model directory
   * @param {object} options - Loading options
   * @param {function} options.progressCallback - Progress callback
   * @param {string} options.device - Device to use ('webgpu' or 'wasm')
   */
  async load(modelPath, options = {}) {
    const { progressCallback, device = 'webgpu' } = options;

    const report = (status, progress = 0, file = '') => {
      if (progressCallback) {
        progressCallback({ status, progress, file });
      }
    };

    // Determine execution provider
    const executionProviders = device === 'webgpu'
      ? ['webgpu', 'wasm']
      : ['wasm'];

    try {
      // Load tokenizer using transformers.js
      report('loading', 0, 'tokenizer');
      this.tokenizer = await AutoTokenizer.from_pretrained(modelPath);

      // Load chat template if not set
      if (!this.tokenizer.chat_template) {
        try {
          const templateResponse = await fetch(`${modelPath}/chat_template.jinja`);
          if (templateResponse.ok) {
            const template = await templateResponse.text();
            this.tokenizer.chat_template = template;
            console.log('Loaded chat template from file');
          }
        } catch (e) {
          console.warn('Could not load chat template:', e);
        }
      }

      // Find special token IDs
      // Try multiple locations where vocab might be stored
      const vocab = this.tokenizer.model?.vocab ||
                    this.tokenizer.model?.tokens_to_ids ||
                    {};

      // Try to get image token ID
      if (vocab instanceof Map) {
        this.imageTokenId = vocab.get('<image>');
        this.eosTokenId = vocab.get('<|im_end|>');
      } else {
        this.imageTokenId = vocab['<image>'];
        this.eosTokenId = vocab['<|im_end|>'];
      }

      // Fallback: encode the token to get its ID
      if (!this.imageTokenId) {
        try {
          const encoded = this.tokenizer.encode('<image>');
          if (encoded && encoded.length > 0) {
            this.imageTokenId = encoded[0];
          }
        } catch (e) {
          console.warn('Could not encode <image> token');
        }
      }

      // Fallback for EOS
      if (!this.eosTokenId) {
        this.eosTokenId = this.tokenizer.eos_token_id;
      }

      console.log('Image token ID:', this.imageTokenId);
      console.log('EOS token ID:', this.eosTokenId);

      // Load config
      report('loading', 10, 'config');
      const configResponse = await fetch(`${modelPath}/config.json`);
      this.config = await configResponse.json();
      // VL models have config in text_config
      const textConfig = this.config.text_config || this.config;
      this.hiddenSize = textConfig.hidden_size || 1024;
      this.numKVHeads = textConfig.num_key_value_heads || 8;
      this.headDim = Math.floor(this.hiddenSize / (textConfig.num_attention_heads || 16));
      console.log('Model hidden size:', this.hiddenSize);
      console.log('Model num_kv_heads:', this.numKVHeads);
      console.log('Model head_dim:', this.headDim);

      // Helper to load ONNX model with external data
      const loadOnnxWithExternalData = async (name, progress) => {
        report('loading', progress, `${name}.onnx`);

        const onnxPath = `${modelPath}/onnx/${name}.onnx`;
        const dataPath = `${modelPath}/onnx/${name}.onnx_data`;

        console.log(`Loading ${name}...`);

        // Check if external data file exists and get its size
        let externalDataSize = 0;
        try {
          const headResponse = await fetch(dataPath, { method: 'HEAD' });
          if (headResponse.ok) {
            externalDataSize = parseInt(headResponse.headers.get('content-length') || '0', 10);
          }
        } catch (e) {
          externalDataSize = 0;
        }

        const sessionOptions = {
          executionProviders,
        };

        // If external data exists, provide path for ONNX Runtime to fetch it
        if (externalDataSize > 0) {
          console.log(`${name} has external data (${(externalDataSize / 1024 / 1024).toFixed(1)} MB)`);

          // Use externalData with path - ONNX Runtime Web will fetch it
          sessionOptions.externalData = [{
            path: `${name}.onnx_data`,
            data: dataPath,  // URL string - ONNX Runtime will fetch
          }];

          console.log(`Creating session for ${name} from URL with external data path...`);
          try {
            const session = await ort.InferenceSession.create(onnxPath, sessionOptions);
            console.log(`Session created for ${name}`);
            return session;
          } catch (e) {
            console.error(`Failed to create session for ${name} with URL approach:`, e);

            // Fallback: fetch both files manually
            console.log(`Fallback: fetching ${name} files manually...`);
            const onnxResponse = await fetch(onnxPath);
            const onnxBuffer = await onnxResponse.arrayBuffer();
            console.log(`Loaded ${name}.onnx: ${onnxBuffer.byteLength} bytes`);

            const dataResponse = await fetch(dataPath);
            const dataBuffer = await dataResponse.arrayBuffer();
            console.log(`Loaded ${name}.onnx_data: ${dataBuffer.byteLength} bytes`);

            sessionOptions.externalData = [{
              path: `${name}.onnx_data`,
              data: new Uint8Array(dataBuffer),
            }];

            const session = await ort.InferenceSession.create(onnxBuffer, sessionOptions);
            console.log(`Session created for ${name} (fallback)`);
            return session;
          }
        } else {
          // No external data, load directly from URL
          console.log(`Loading ${name} directly from URL (no external data)...`);
          try {
            const session = await ort.InferenceSession.create(onnxPath, sessionOptions);
            console.log(`Session created for ${name}`);
            return session;
          } catch (e) {
            console.error(`Failed to create session for ${name}:`, e);
            throw new Error(`Failed to load ${name}: ${e.message || e}`);
          }
        }
      };

      // Load embed_tokens
      this.embedTokensSession = await loadOnnxWithExternalData('embed_tokens', 20);

      // Load embed_images
      this.embedImagesSession = await loadOnnxWithExternalData('embed_images', 40);

      // Load decoder
      this.decoderSession = await loadOnnxWithExternalData('decoder', 60);

      report('done', 100, '');
      return true;

    } catch (error) {
      console.error('Failed to load VL model:', error);
      throw error;
    }
  }

  /**
   * Process images and get embeddings
   * @param {string[]} imageUrls - Array of image URLs or data URLs
   * @returns {Promise<{embeddings: Float32Array, numTokens: number}>}
   */
  async getImageEmbeddings(imageUrls) {
    const allEmbeddings = [];
    let totalTokens = 0;

    for (const url of imageUrls) {
      // Load and process image
      const img = await loadImage(url);
      const processed = await processImage(img);

      console.log('Image processed:', {
        numTiles: processed.numTiles,
        shape: processed.shape,
      });

      // Create tensors
      const pixelValuesTensor = new ort.Tensor(
        'float32',
        processed.pixelValues,
        processed.shape  // [num_tiles, 1024, 768]
      );

      const attentionMaskTensor = new ort.Tensor(
        'int64',
        processed.attentionMask,  // BigInt64Array
        [processed.numTiles, 1024]  // [num_tiles, patches_per_tile]
      );

      // Run embed_images
      const outputs = await this.embedImagesSession.run({
        pixel_values: pixelValuesTensor,
        patch_attention_mask: attentionMaskTensor,
      });

      // Output shape: [num_tiles, num_image_tokens, hidden_dim]
      const embeddings = outputs.image_embeddings;
      console.log('Image embeddings shape:', embeddings.dims);

      // Flatten to [total_tokens, hidden_dim]
      const [numTiles, tokensPerTile, hiddenDim] = embeddings.dims;
      totalTokens += numTiles * tokensPerTile;
      allEmbeddings.push(embeddings.data);
    }

    // Concatenate all image embeddings
    const totalLength = allEmbeddings.reduce((sum, e) => sum + e.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const emb of allEmbeddings) {
      combined.set(emb, offset);
      offset += emb.length;
    }

    return { embeddings: combined, numTokens: totalTokens };
  }

  /**
   * Get text embeddings from token IDs
   * @param {number[]} inputIds - Token IDs as regular numbers
   * @returns {Promise<ort.Tensor>} - Text embeddings tensor
   */
  async getTextEmbeddings(inputIds) {
    const inputTensor = new ort.Tensor(
      'int64',
      new BigInt64Array(inputIds.map(id => BigInt(id))),
      [1, inputIds.length]
    );
    const outputs = await this.embedTokensSession.run({ input_ids: inputTensor });
    return outputs.inputs_embeds;
  }

  /**
   * Build combined embeddings by replacing image tokens with image embeddings
   */
  buildCombinedEmbeddings(inputIds, textEmbeddings, imageEmbeddings, numImageTokens) {
    const [, seqLen, hiddenDim] = textEmbeddings.dims;
    const textEmb = textEmbeddings.data;
    const imgEmb = imageEmbeddings;

    // Find image token positions
    const imagePositions = [];
    for (let i = 0; i < inputIds.length; i++) {
      if (inputIds[i] === this.imageTokenId) {
        imagePositions.push(i);
      }
    }

    if (imagePositions.length === 0 || numImageTokens === 0) {
      return textEmbeddings;
    }

    // Each image token gets replaced with numImageTokens/numImages embeddings
    const tokensPerImage = Math.floor(numImageTokens / imagePositions.length);

    // Calculate output size
    const outputSeqLen = seqLen - imagePositions.length + numImageTokens;
    const result = new Float32Array(outputSeqLen * hiddenDim);

    let srcTextIdx = 0;
    let dstIdx = 0;
    let imgOffset = 0;

    for (let i = 0; i < seqLen; i++) {
      if (imagePositions.includes(i)) {
        // Copy image embeddings for this image token
        const copyLen = tokensPerImage * hiddenDim;
        result.set(imgEmb.slice(imgOffset, imgOffset + copyLen), dstIdx * hiddenDim);
        imgOffset += copyLen;
        dstIdx += tokensPerImage;
      } else {
        // Copy text embedding
        const start = i * hiddenDim;
        result.set(textEmb.slice(start, start + hiddenDim), dstIdx * hiddenDim);
        dstIdx++;
      }
    }

    return new ort.Tensor('float32', result, [1, outputSeqLen, hiddenDim]);
  }

  /**
   * Initialize cache for decoder (both conv states and KV cache)
   */
  initializeCache() {
    const cache = {};

    for (const name of this.decoderSession.inputNames) {
      if (name.startsWith('past_conv')) {
        // Conv states: [batch, hidden_size, kernel_size-1]
        // Kernel size is 4, so we need 3 states
        cache[name] = new ort.Tensor(
          'float32',
          new Float32Array(1 * this.hiddenSize * 3),
          [1, this.hiddenSize, 3]
        );
      } else if (name.startsWith('past_key_values')) {
        // KV cache: [batch, num_kv_heads, past_seq_len, head_dim]
        // Initialize with 0 length sequence
        cache[name] = new ort.Tensor(
          'float32',
          new Float32Array(0),  // Empty cache initially
          [1, this.numKVHeads, 0, this.headDim]
        );
      }
    }

    return cache;
  }

  /**
   * Update cache from decoder outputs
   */
  updateCache(cache, outputs) {
    for (const name of Object.keys(outputs)) {
      if (name.startsWith('present_conv')) {
        // Conv states: present_conv.X -> past_conv.X
        const cacheName = name.replace('present_conv', 'past_conv');
        if (cacheName in cache) {
          cache[cacheName] = outputs[name];
        }
      } else if (name.startsWith('present.')) {
        // KV cache: present.X.key -> past_key_values.X.key
        const cacheName = name.replace('present.', 'past_key_values.');
        if (cacheName in cache) {
          cache[cacheName] = outputs[name];
        }
      }
    }
  }

  /**
   * Generate text given messages with optional images
   * @param {Array} messages - Chat messages
   * @param {object} options - Generation options
   */
  async generate(messages, options = {}) {
    const { maxNewTokens = 256, onToken, images = [] } = options;

    console.log('=== VL Generate ===');
    console.log('Messages:', JSON.stringify(messages, null, 2));
    console.log('Images count:', images.length);

    // Build prompt with chat template
    // For VL models, we need to insert <image> tokens for each image
    let promptMessages = messages;
    if (images.length > 0) {
      // Prepend image tokens to the first user message
      promptMessages = messages.map((msg, idx) => {
        if (msg.role === 'user' && idx === 0) {
          const imageTokens = images.map(() => '<image>').join('');
          return { ...msg, content: imageTokens + msg.content };
        }
        return msg;
      });
    }

    console.log('Prompt messages with image tokens:', JSON.stringify(promptMessages, null, 2));

    // Apply chat template
    const prompt = this.tokenizer.apply_chat_template(promptMessages, {
      add_generation_prompt: true,
      tokenize: false,
    });

    console.log('Full prompt:', prompt);

    // Tokenize
    const encoded = this.tokenizer.encode(prompt);
    const inputIds = [...encoded];  // Convert to regular array

    console.log('Input IDs length:', inputIds.length);
    console.log('Image token ID:', this.imageTokenId);

    // Count image tokens in input
    const imageTokenCount = inputIds.filter(id => id === this.imageTokenId).length;
    console.log('Image tokens in input:', imageTokenCount);

    // Get text embeddings
    const textEmbeddings = await this.getTextEmbeddings(inputIds);
    console.log('Text embeddings shape:', textEmbeddings.dims);

    // Get image embeddings if images provided
    let inputsEmbeds;
    if (images.length > 0) {
      const { embeddings: imageEmbeddings, numTokens } = await this.getImageEmbeddings(images);
      console.log('Image embeddings length:', imageEmbeddings.length);
      console.log('Total image tokens from embeddings:', numTokens);
      console.log('Hidden dim:', this.hiddenSize);
      console.log('Tokens per image:', numTokens / images.length);
      inputsEmbeds = this.buildCombinedEmbeddings(inputIds, textEmbeddings, imageEmbeddings, numTokens);
    } else {
      inputsEmbeds = textEmbeddings;
    }

    console.log('Combined embeddings shape:', inputsEmbeds.dims);

    // Initialize cache
    const cache = this.initializeCache();

    // Generation loop
    let curLen = inputsEmbeds.dims[1];
    const generatedTokens = [];
    let embeds = inputsEmbeds;

    for (let step = 0; step < maxNewTokens; step++) {
      // Prepare attention mask
      const attentionMask = new ort.Tensor(
        'int64',
        new BigInt64Array(curLen).fill(1n),
        [1, curLen]
      );

      // Prepare position IDs
      let positionIds;
      if (step === 0) {
        positionIds = new ort.Tensor(
          'int64',
          new BigInt64Array(embeds.dims[1]).map((_, i) => BigInt(i)),
          [1, embeds.dims[1]]
        );
      } else {
        positionIds = new ort.Tensor(
          'int64',
          [BigInt(curLen - 1)],
          [1, 1]
        );
      }

      // Run decoder
      const feeds = {
        inputs_embeds: embeds,
        attention_mask: attentionMask,
        position_ids: positionIds,
        ...cache,
      };

      const outputs = await this.decoderSession.run(feeds);

      // Get logits - shape is [batch, seq_len, vocab_size]
      const logits = outputs.logits;
      const vocabSize = logits.dims[2];
      const logitsData = logits.data;

      // Get last token logits
      const lastLogitStart = (logits.dims[1] - 1) * vocabSize;
      const lastLogits = logitsData.slice(lastLogitStart, lastLogitStart + vocabSize);

      // Greedy decoding - find max
      let maxIdx = 0;
      let maxVal = lastLogits[0];
      for (let i = 1; i < vocabSize; i++) {
        if (lastLogits[i] > maxVal) {
          maxVal = lastLogits[i];
          maxIdx = i;
        }
      }

      generatedTokens.push(maxIdx);

      // Callback with token
      if (onToken) {
        const tokenText = this.tokenizer.decode([maxIdx]);
        const shouldStop = onToken(tokenText, maxIdx);
        if (shouldStop) break;
      }

      // Check for EOS
      if (maxIdx === this.eosTokenId) {
        break;
      }

      // Update cache
      this.updateCache(cache, outputs);

      // Get embedding for next token
      const nextEmbeds = await this.getTextEmbeddings([maxIdx]);
      embeds = nextEmbeds;
      curLen++;
    }

    return this.tokenizer.decode(generatedTokens, { skip_special_tokens: true });
  }

  /**
   * Free resources
   */
  dispose() {
    // ONNX Runtime Web sessions are automatically cleaned up
    this.tokenizer = null;
    this.embedTokensSession = null;
    this.embedImagesSession = null;
    this.decoderSession = null;
  }
}

export default VLModel;
