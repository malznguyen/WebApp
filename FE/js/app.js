function app() {
    return {
        // ===== STATE MANAGEMENT =====
        activeTab: 'image-search',
        statusMessage: 'Connecting...',
        isLoading: false,
        loadingTitle: '',
        loadingMessage: '',
        searchTerm: '',
        backendConnected: false,
        backendConnectionStatus: 'connecting',

        // ===== NEW VISION STATE =====
        visionCapabilities: {
            available: false,
            apiConfigured: false,
            supportedLanguages: ['vietnamese', 'english'],
            detailLevels: ['brief', 'detailed', 'extensive']
        },

        visionSettings: {
            language: 'vietnamese',
            detailLevel: 'detailed'
        },

        visionState: {
            isAnalyzing: false,
            progress: 0,
            status: '',
            result: null,
            taskId: null
        },

        detectionSettings: {},

        detectionState: {
            isAnalyzing: false,
            progress: 0,
            status: '',
            result: null,
            taskId: null
        },

        metadataSettings: {},

        metadataState: {
            isAnalyzing: false,
            result: null,
            error: null
        },

        selectedActions: {
            vision: false,
            detection: false,
            metadata: false,
            search: false
        },

        settingsVisibility: {
            vision: false,
            search: false
        },

        // ===== IMAGE SEARCH STATE =====
        searchOptions: {
            socialOnly: false
        },
        isSearching: false,
        searchProgress: 0,
        searchStatus: '',
        searchResults: [],
        activeSearchId: null,
        resultsFilter: 'all',
        viewMode: 'list',
        canSearch: false,
        selectedImage: null,

        // ===== DOCUMENT PROCESSING STATE =====
        processingMode: 'individual',

        // 🔥 NEW SUMMARY MODE STATE - THE HEART OF OUR REFACTOR! 
        summaryMode: 'full', // 'full' | 'percentage' | 'word-count'
        detailLevel: 50, // Only used when summaryMode === 'percentage'
        wordCountLimit: 500, // Only used when summaryMode === 'word-count'

        targetLanguage: null,
        isProcessing: false,
        processingQueue: [],
        activePreviewFileId: null,
        originalContent: '',
        aiResult: null,
        analysisData: null,
        contentTab: 'original',
        canProcess: false,

        // Updated panels state với summaryMode panel
        settingsPanels: {
            summaryMode: true,    // 🆕 Mở mặc định để user thấy ngay
            processing: false,
            language: false,
            queue: true,
            advanced: false
        },

        urlInput: '',
        activeProcessIdToItemId: {},

        // ===== UI STATE =====
        logPanelExpanded: false,
        logEntries: [],
        notification: {
            show: false,
            type: 'info',
            title: '',
            message: ''
        },
        stats: {
            processed: 0,
            avgTime: '0s'
        },

        // ===== INITIALIZATION =====
        async initApp() {
            console.log('🚀 Enhanced Toolkit v2.0 - Initializing Alpine.js app...');
            this.addLogEntry('info', 'Application initializing...');

            // Initialize watchers for summary mode changes
            this.initializeSummaryModeWatchers();

            if (typeof eel !== 'undefined') {
                eel.expose(this.handleEelProcessingComplete.bind(this), 'processingComplete');
                eel.expose(this.handleEelProcessingError.bind(this), 'processingError');
                eel.expose(this.handleEelDocProcessingProgress.bind(this), 'processingProgress');

                eel.expose(this.handleEelSearchProgress.bind(this), 'searchProgress');
                eel.expose(this.handleEelSearchComplete.bind(this), 'searchComplete');
                eel.expose(this.handleEelSearchError.bind(this), 'searchError');

                // Vision callbacks
                eel.expose(this.handleVisionProgress.bind(this), 'visionProgress');
                eel.expose(this.handleVisionComplete.bind(this), 'visionComplete');
                eel.expose(this.handleVisionError.bind(this), 'visionError');

                eel.expose(this.handleDetectionProgress.bind(this), 'detectionProgress');
                eel.expose(this.handleDetectionComplete.bind(this), 'detectionComplete');
                eel.expose(this.handleDetectionError.bind(this), 'detectionError');

                this.addLogEntry('info', 'Eel callbacks exposed to Python.');
            } else {
                this.addLogEntry('warning', 'Eel is not defined. App will run in offline/demo mode.');
            }

            this.setupEventListeners();
            await this.connectToBackend();
        },

        // 🆕 NEW METHOD: Initialize watchers for summary mode changes
        initializeSummaryModeWatchers() {
            // Watch for summary mode changes để update UI và validate
            this.$watch('summaryMode', (newMode, oldMode) => {
                this.addLogEntry('info', `Summary mode changed: ${oldMode} → ${newMode}`);

                // Reset irrelevant values để tránh confusion
                if (newMode === 'full') {
                    // Full mode không cần parameters
                    this.addLogEntry('debug', 'Full summary mode activated - no parameters needed');
                } else if (newMode === 'percentage' && oldMode !== 'percentage') {
                    // Ensure valid percentage range
                    if (this.detailLevel < 10 || this.detailLevel > 90) {
                        this.detailLevel = 50; // Safe default
                        this.addLogEntry('warning', 'Reset detail level to 50% due to invalid value');
                    }
                } else if (newMode === 'word-count' && oldMode !== 'word-count') {
                    // Ensure valid word count
                    if (this.wordCountLimit < 50 || this.wordCountLimit > 5000) {
                        this.wordCountLimit = 500; // Safe default
                        this.addLogEntry('warning', 'Reset word count limit to 500 due to invalid value');
                    }
                }
            });

            // Watch for detail level changes (percentage mode)
            this.$watch('detailLevel', (newVal) => {
                if (this.summaryMode === 'percentage') {
                    // Clamp value để đảm bảo an toàn
                    if (newVal < 10) this.detailLevel = 10;
                    else if (newVal > 90) this.detailLevel = 90;
                }
            });

            // Watch for word count changes
            this.$watch('wordCountLimit', (newVal) => {
                if (this.summaryMode === 'word-count') {
                    // Validate và clamp
                    if (newVal < 50) this.wordCountLimit = 50;
                    else if (newVal > 5000) this.wordCountLimit = 5000;
                    else if (isNaN(newVal)) this.wordCountLimit = 500; // Fallback for NaN
                }
            });
        },

        async connectToBackend(retries = 4, delay = 500) {
            for (let i = 0; i < retries; i++) {
                try {
                    this.backendConnectionStatus = `Connecting... (Attempt ${i + 1})`;
                    this.addLogEntry('info', `Attempting to connect to Python backend... (Attempt ${i + 1}/${retries})`);

                    if (typeof eel !== 'undefined' && typeof eel.get_app_config === 'function') {
                        await this.loadAppConfig();
                        await this.initVisionCapabilities();

                        this.backendConnected = true;
                        this.backendConnectionStatus = 'Connected';
                        this.statusMessage = 'Ready';
                        this.addLogEntry('success', 'Successfully connected to Python backend.');

                        return;
                    }
                    throw new Error("Eel functions not available yet.");
                } catch (error) {
                    this.addLogEntry('warning', `Backend connection attempt ${i + 1} failed: ${error.message}`);
                    if (i === retries - 1) {
                        this.backendConnected = false;
                        this.backendConnectionStatus = 'Connection Failed';
                        this.statusMessage = 'Error: Connection Failed';
                        this.addLogEntry('error', 'All attempts to connect to backend failed. Using mock data for fallback.');
                        this.showNotification('error', 'Connection Failed', 'Could not connect to the Python backend. The app will run in a limited, offline mode.');
                        this.useMockDataAsFallback();
                    } else {
                        await this.delay(delay * (i + 1));
                    }
                }
            }
        },

        useMockDataAsFallback() {
            this.addLogEntry('warning', 'Using mock configuration for demo mode due to connection failure.');
            this.getMockConfig();
            this.canSearch = true;
            this.visionCapabilities.available = true;
            this.visionCapabilities.apiConfigured = true;
        },

        async loadAppConfig() {
            this.addLogEntry('info', 'Loading application configuration from backend...');
            const config = await eel.get_app_config()();
            if (config && config.status === 'ready') {
                this.addLogEntry('info', `Config loaded - SERP:${config.has_serp_api}, Imgur:${config.has_imgur}, ChatGPT:${config.has_chatgpt}, Vision:${config.has_vision}`);
                this.canSearch = !!(config.has_serp_api && config.has_imgur);
                this.visionCapabilities.available = !!config.vision_available;
                this.visionCapabilities.apiConfigured = !!config.has_vision;
            } else {
                throw new Error(config.error || 'Failed to load valid config from backend.');
            }
        },

        getMockConfig() {
            this.addLogEntry('info', 'Setting mock configuration.');
        },

        setupEventListeners() {
            this.addLogEntry('info', 'Setting up event listeners.');
            document.addEventListener('dragover', (e) => { e.preventDefault(); e.stopPropagation(); });
            document.addEventListener('drop', (e) => {
                e.preventDefault(); e.stopPropagation();
                if (this.activeTab === 'image-search') this.handleImageDrop(e);
                else if (this.activeTab === 'document-summary') this.handleDocumentDrop(e);
            });
            document.addEventListener('keydown', (e) => {
                if (e.ctrlKey || e.metaKey) {
                    const key = e.key.toLowerCase();
                    if (key === '1') { e.preventDefault(); this.activeTab = 'image-search'; }
                    else if (key === '2') { e.preventDefault(); this.activeTab = 'document-summary'; }
                    else if (key === 'l') { e.preventDefault(); this.logPanelExpanded = !this.logPanelExpanded; }
                }
            });
        },

        // ===== IMAGE SEARCH METHODS (không đổi) =====
        handleImageDrop(event) {
            const files = Array.from(event.dataTransfer.files);
            const imageFiles = files.filter(file => file.type.startsWith('image/'));
            if (imageFiles.length > 0) {
                this.addLogEntry('info', `Image dropped: ${imageFiles[0].name}`);
                this.processImageFile(imageFiles[0]);
            } else {
                this.showNotification('error', 'Invalid File Type', 'Please drop an image file (e.g., JPG, PNG).');
                this.addLogEntry('warning', 'Invalid file dropped for image search.');
            }
        },

        async processImageFile(file) {
            this.addLogEntry('info', `Processing image: ${file.name} (${this.formatFileSize(file.size)})`);
            this.isLoading = true;
            this.loadingTitle = 'Processing Image';
            this.loadingMessage = 'Validating and preparing image...';
            try {
                const MAX_IMAGE_SIZE = 10 * 1024 * 1024;
                if (file.size > MAX_IMAGE_SIZE) throw new Error(`File too large. Max size: ${this.formatFileSize(MAX_IMAGE_SIZE)}.`);
                if (!['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'].includes(file.type)) {
                    throw new Error('Invalid image type. Supported: JPG, PNG, GIF, WEBP, BMP.');
                }
                const base64 = await this.fileToBase64(file);
                this.selectedImage = { name: file.name, size: file.size, data: base64, preview: base64 };
                this.searchResults = [];
                this.addLogEntry('success', `Image '${file.name}' loaded and ready for search.`);
                this.showNotification('success', 'Image Loaded', `${file.name} is ready to search.`);
            } catch (error) {
                this.addLogEntry('error', `Image processing failed for ${file.name}: ${error.message}`);
                this.showNotification('error', 'Image Processing Error', error.message);
                this.selectedImage = null;
            } finally {
                this.isLoading = false;
            }
        },

        browseImageFiles() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/jpeg,image/png,image/gif,image/webp,image/bmp';
            input.onchange = (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.addLogEntry('info', `Image file selected via browse: ${e.target.files[0].name}`);
                    this.processImageFile(e.target.files[0]);
                }
            };
            input.click();
        },

        removeSelectedImage() {
            if (this.selectedImage) this.addLogEntry('info', `Removing selected image: ${this.selectedImage.name}`);
            this.selectedImage = null;
            this.searchResults = [];
            this.searchProgress = 0;
            this.searchStatus = '';
            this.clearVisionResult();
            this.metadataState.result = null;
            this.metadataState.error = null;
            this.metadataState.isAnalyzing = false;
        },

        async runSelectedActions() {
            const actions = [];
            if (this.selectedActions.vision) actions.push(this.analyzeImageWithVision());
            if (this.selectedActions.detection) actions.push(this.analyzeImageForAI());
            if (this.selectedActions.metadata) actions.push(this.analyzeImageMetadata());
            if (this.selectedActions.search) actions.push(this.startImageSearch());
            if (actions.length) await Promise.all(actions);
        },

        async startImageSearch() {
            if (!this.selectedImage || !this.canSearch || this.isSearching) {
                if (!this.canSearch && this.selectedImage) {
                    this.showNotification('warning', 'APIs Missing', 'Image search requires SERP API and Imgur Client ID. Please check configuration.');
                }
                return;
            }
            this.isSearching = true;
            this.searchProgress = 0;
            this.searchResults = [];
            this.searchStatus = 'Initializing image search...';
            this.addLogEntry('info', `Starting image search for '${this.selectedImage.name}'. Social only: ${this.searchOptions.socialOnly}.`);
            try {
                if (this.backendConnected) {
                    const searchId = await eel.search_image_async_web(this.selectedImage.data, this.selectedImage.name, this.searchOptions.socialOnly)();
                    this.activeSearchId = searchId;
                    this.addLogEntry('info', `Async image search started with ID: ${searchId}`);
                } else {
                    this.addLogEntry('info', 'Running image search in demo mode.');
                    await this.runDemoSearch();
                    this.isSearching = false;
                }
            } catch (error) {
                const errorMsg = `Image search initiation failed for ${this.selectedImage.name}: ${error.message || 'Unknown error'}`;
                this.addLogEntry('error', errorMsg);
                this.showNotification('error', 'Search Start Failed', error.message || 'An unknown error occurred.');
                this.searchStatus = `Error: ${error.message || 'Unknown error'}`;
                this.isSearching = false;
                this.searchProgress = 0;
            }
        },

        async runDemoSearch() {
            const steps = [
                { progress: 10, message: 'Demo: Validating image...' },
                { progress: 20, message: 'Demo: Uploading to Imgur...' },
                { progress: 40, message: 'Demo: Analyzing image with search engine...' },
                { progress: 60, message: 'Demo: Querying search engines...' },
                { progress: 80, message: 'Demo: Processing and filtering results...' },
                { progress: 100, message: 'Demo: Search complete.' }
            ];
            for (const step of steps) {
                await this.delay(500 + Math.random() * 300);
                this.handleEelSearchProgress('demo_search_id', step.progress, step.message);
            }
            this.handleEelSearchComplete('demo_search_id', this.getMockSearchResults());
        },

        handleEelSearchProgress(searchId, percent, message) {
            this.searchProgress = Math.max(0, Math.min(100, percent));
            this.searchStatus = message;
            this.addLogEntry('info', `[Search PID: ${searchId}] ${percent}% - ${message}`);
        },

        handleEelSearchComplete(searchId, results) {
            this.searchResults = results || [];
            this.addLogEntry('success', `Async image search (ID: ${searchId}) successful. Found ${this.searchResults.length} results.`);
            this.showNotification('success', 'Search Complete', `Found ${this.searchResults.length} results.`);
            this.searchStatus = `Search completed. Found ${this.searchResults.length} results.`;
            this.isSearching = false;
            this.searchProgress = 100;
            this.activeSearchId = null;
        },

        handleEelSearchError(searchId, errorTitle, errorMessage) {
            this.addLogEntry('error', `Async image search (ID: ${searchId}) failed: ${errorTitle} - ${errorMessage}`);
            this.showNotification('error', errorTitle, errorMessage);
            this.searchResults = [];
            this.searchStatus = `Error: ${errorMessage}`;
            this.isSearching = false;
            this.searchProgress = 0;
            this.activeSearchId = null;
        },

        // ===== VISION METHODS (không đổi) =====
        async initVisionCapabilities() {
            try {
                if (this.backendConnected) {
                    const capabilities = await eel.get_vision_capabilities()();
                    this.visionCapabilities = { ...this.visionCapabilities, ...capabilities };

                    this.addLogEntry('info', `Vision capabilities loaded. Available: ${capabilities.available}, Configured: ${capabilities.api_configured}`);

                    if (!capabilities.available) {
                        this.addLogEntry('warning', 'Vision module not available. Install openai>=1.10.0');
                    } else if (!capabilities.api_configured) {
                        this.addLogEntry('warning', 'OpenAI API key not configured for Vision analysis');
                    }
                } else {
                    this.addLogEntry('warning', 'Skipping vision capabilities check, backend not connected.');
                }
            } catch (error) {
                this.addLogEntry('error', `Failed to load vision capabilities: ${error.message}`);
                this.visionCapabilities.available = false;
                this.visionCapabilities.apiConfigured = false;
            }
        },

        async analyzeImageWithVision() {
            if (!this.selectedImage || !this.visionCapabilities.available || this.visionState.isAnalyzing) {
                if (!this.visionCapabilities.available) {
                    this.showNotification('warning', 'Vision Not Available', 'Vision analysis requires a configured backend.');
                }
                if (!this.visionCapabilities.apiConfigured) {
                    this.showNotification('warning', 'Vision Not Ready', 'Vision API Key is not configured on the backend.');
                }
                return;
            }

            this.visionState.isAnalyzing = true;
            this.visionState.progress = 0;
            this.visionState.result = null;
            this.visionState.status = 'Preparing image for analysis...';

            this.addLogEntry('info', `Starting vision analysis for '${this.selectedImage.name}' (${this.visionSettings.language}, ${this.visionSettings.detailLevel})`);

            try {
                if (this.backendConnected) {
                    const taskId = await eel.describe_image_async_web(
                        this.selectedImage.data,
                        this.selectedImage.name,
                        this.visionSettings.language,
                        this.visionSettings.detailLevel
                    )();

                    this.visionState.taskId = taskId;
                    this.addLogEntry('info', `Vision analysis started with task ID: ${taskId}`);
                } else {
                    await this.runDemoVisionAnalysis();
                }
            } catch (error) {
                this.handleVisionError('demo_task_id', 'Vision Analysis Failed', error.message || 'Unknown error occurred');
            }
        },

        async runDemoVisionAnalysis() {
            const steps = [
                { progress: 10, message: 'Demo: Validating image...' },
                { progress: 30, message: 'Demo: Sending to OpenAI Vision...' },
                { progress: 60, message: 'Demo: Analyzing image content...' },
                { progress: 90, message: 'Demo: Generating description...' },
                { progress: 100, message: 'Demo: Analysis complete!' }
            ];

            for (const step of steps) {
                await this.delay(800 + Math.random() * 400);
                this.handleVisionProgress('demo_task', step.progress, step.message);
            }

            const mockResult = {
                success: true,
                description: `[DEMO] This image shows a ${this.selectedImage.name.includes('car') ? 'car' : 'subject'} in a scenic setting. The colors are vibrant and the composition is well-balanced, creating an interesting visual experience.`,
                language: this.visionSettings.language,
                detail_level: this.visionSettings.detailLevel,
                text_metrics: {
                    word_count: 45,
                    char_count: 198,
                    sentence_count: 3
                },
                processing_time_seconds: 2.3,
                api_usage: {
                    cost_estimate: 0.005,
                    total_tokens: 150
                },
                image_metadata: {
                    format: 'JPEG',
                    size: [800, 600],
                    file_size_mb: 1.2
                },
                filename: this.selectedImage.name
            };

            this.handleVisionComplete('demo_task', mockResult);
        },

        handleVisionProgress(taskId, percent, message) {
            if (this.visionState.taskId === taskId || taskId.startsWith('demo')) {
                this.visionState.progress = Math.max(0, Math.min(100, percent));
                this.visionState.status = message;
                this.addLogEntry('info', `[Vision ${taskId}] ${percent}% - ${message}`);
            }
        },

        handleVisionComplete(taskId, result) {
            if (this.visionState.taskId === taskId || taskId.startsWith('demo')) {
                this.visionState.result = ImageVision.formatVisionResult(result);
                this.visionState.isAnalyzing = false;
                this.visionState.progress = 100;
                this.visionState.status = 'Analysis completed successfully';

                const wordCount = result.text_metrics?.word_count || 0;
                const cost = result.api_usage?.cost_estimate || 0;

                this.addLogEntry('success', `Vision analysis completed for '${result.filename}': ${wordCount} words, $${cost.toFixed(4)}`);
                this.showNotification('success', 'Vision Analysis Complete', `Generated ${wordCount} word description for ${result.filename}`);
            }
        },

        handleVisionError(taskId, errorTitle, errorMessage) {
            this.visionState.isAnalyzing = false;
            this.visionState.progress = 0;
            this.visionState.status = `Error: ${errorMessage}`;
            this.visionState.result = { error: errorMessage, description: null };

            this.addLogEntry('error', `Vision analysis failed: ${errorTitle} - ${errorMessage}`);
            this.showNotification('error', errorTitle, errorMessage);
        },

        copyVisionDescription() {
            if (this.visionState.result && this.visionState.result.description) {
                Common.copyToClipboard(this.visionState.result.description)
                    .then(() => {
                        this.showNotification('success', 'Copied', 'Vision description copied to clipboard');
                        this.addLogEntry('info', 'Vision description copied to clipboard');
                    })
                    .catch(error => {
                        this.showNotification('error', 'Copy Failed', 'Could not copy to clipboard');
                        this.addLogEntry('error', `Copy failed: ${error.message}`);
                    });
            }
        },

        exportVisionResult() {
            if (this.visionState.result && this.visionState.result.description) {
                const filename = this.selectedImage?.name?.replace(/\.[^/.]+$/, "") || 'vision_analysis';
                ImageVision.exportVisionResult(this.visionState.result, filename);
                this.addLogEntry('info', `Vision result exported for ${filename}`);
            }
        },

        clearVisionResult() {
            this.visionState.result = null;
            this.visionState.progress = 0;
            this.visionState.status = '';
            this.addLogEntry('info', 'Vision analysis result cleared');
        },

        async analyzeImageMetadata() {
            if (!this.selectedImage || this.metadataState.isAnalyzing) return;
            this.metadataState.isAnalyzing = true;
            this.metadataState.error = null;
            this.metadataState.result = null;
            try {
                if (this.backendConnected) {
                    const res = await eel.analyze_image_metadata(this.selectedImage.data, this.selectedImage.name, true)();
                    if (res && res.success) {
                        this.metadataState.result = ImageMetadata.formatForDisplay(res);
                        this.addLogEntry('info', `Metadata extracted for ${this.selectedImage.name}`);
                    } else {
                        this.metadataState.error = res.error || 'Metadata extraction failed';
                        this.addLogEntry('error', `Metadata extraction failed: ${this.metadataState.error}`);
                    }
                } else {
                    this.metadataState.error = 'Backend not connected';
                }
            } catch (err) {
                this.metadataState.error = err.message || 'Error';
            } finally {
                this.metadataState.isAnalyzing = false;
            }
        },

        exportImageMetadata(format) {
            if (this.metadataState.result) {
                const base = this.selectedImage?.name?.replace(/\.[^/.]+$/, '') || 'image_metadata';
                ImageMetadata.exportMetadata(this.metadataState.result, base, format);
            }
        },

        async analyzeImageForAI() {
            if (!this.selectedImage || this.detectionState.isAnalyzing || !this.visionCapabilities.available) return;
            this.detectionState.isAnalyzing = true;
            this.detectionState.progress = 0;
            this.detectionState.result = null;
            this.detectionState.status = 'Preparing image for detection...';

            this.addLogEntry('info', `Starting AI detection for '${this.selectedImage.name}'`);
            try {
                if (this.backendConnected) {
                    const taskId = await eel.detect_ai_image_async_web(this.selectedImage.data, this.selectedImage.name)();
                    this.detectionState.taskId = taskId;
                    this.addLogEntry('info', `AI detection started with task ID: ${taskId}`);
                } else {
                    await this.runDemoAIDetection();
                }
            } catch (error) {
                this.handleDetectionError('demo_task_id', 'AI Detection Failed', error.message || 'Unknown error');
            }
        },

        async runDemoAIDetection() {
            const steps = [
                { progress: 15, message: 'Demo: Validating...' },
                { progress: 40, message: 'Demo: Sending to AI...' },
                { progress: 80, message: 'Demo: Evaluating...' },
                { progress: 100, message: 'Demo: Done!' }
            ];
            for (const step of steps) {
                await this.delay(700 + Math.random() * 300);
                this.handleDetectionProgress('demo_det', step.progress, step.message);
            }
            const mock = {
                success: true,
                detection: {
                    ai_generated_probability: 65,
                    confidence_level: 'medium',
                    analysis_summary: '[DEMO] Có một số dấu hiệu có thể ảnh do AI tạo ra.',
                    detected_indicators: [
                        { category: 'technical', indicator: 'pixel anomaly', severity: 'moderate', explanation: 'Độ chi tiết không đồng đều' }
                    ],
                    likely_generation_method: 'Unknown'
                },
                filename: this.selectedImage.name,
                processing_time_seconds: 1.2,
                api_usage: { cost_estimate: 0, total_tokens: 0 }
            };
            this.handleDetectionComplete('demo_det', mock);
        },

        handleDetectionProgress(taskId, percent, message) {
            if (this.detectionState.taskId === taskId || taskId.startsWith('demo')) {
                this.detectionState.progress = Math.max(0, Math.min(100, percent));
                this.detectionState.status = message;
                this.addLogEntry('info', `[Detect ${taskId}] ${percent}% - ${message}`);
            }
        },

        handleDetectionComplete(taskId, result) {
            if (this.detectionState.taskId === taskId || taskId.startsWith('demo')) {
                this.detectionState.isAnalyzing = false;
                this.detectionState.progress = 100;
                this.detectionState.status = 'Detection complete';
                this.detectionState.result = ImageDetection.formatDetectionResult(result);
                this.addLogEntry('success', `AI detection completed for '${result.filename}'`);
            }
        },

        handleDetectionError(taskId, title, message) {
            this.detectionState.isAnalyzing = false;
            this.detectionState.progress = 0;
            this.detectionState.status = `Error: ${message}`;
            this.detectionState.result = { error: message };
            this.addLogEntry('error', `AI detection failed: ${title} - ${message}`);
            this.showNotification('error', title, message);
        },

        stopVisionAnalysis() {
            if (this.visionState.taskId) {
                this.visionState.status = 'Stopping...';
                StopDownloadUI.stopTask(this.visionState.taskId);
            }
        },

        stopDetection() {
            if (this.detectionState.taskId) {
                this.detectionState.status = 'Stopping...';
                StopDownloadUI.stopTask(this.detectionState.taskId);
            }
        },

        stopImageSearch() {
            if (this.activeSearchId) {
                this.searchStatus = 'Stopping...';
                StopDownloadUI.stopTask(this.activeSearchId);
            }
        },

        clearDetectionResult() {
            this.detectionState.result = null;
            this.detectionState.progress = 0;
            this.detectionState.status = '';
            this.addLogEntry('info', 'AI detection result cleared');
        },

        // ===== DOCUMENT PROCESSING METHODS (UPDATED FOR SUMMARY MODE) =====
        handleDocumentDrop(event) {
            const files = Array.from(event.dataTransfer.files);
            this.addLogEntry('info', `${files.length} file(s) dropped for document processing.`);
            this.addFilesToQueue(files);
        },

        browseDocumentFiles() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pdf,.doc,.docx,.txt,.md';
            input.multiple = true;
            input.onchange = (e) => {
                if (e.target.files && e.target.files.length > 0) {
                    this.addLogEntry('info', `${e.target.files.length} file(s) selected via browse for document processing.`);
                    this.addFilesToQueue(Array.from(e.target.files));
                }
            };
            input.click();
        },

        async addFilesToQueue(files) {
            this.isLoading = true;
            this.loadingTitle = 'Adding Files';
            let validFilesAddedCount = 0;
            let lastValidFileItem = null;

            for (const file of files) {
                this.loadingMessage = `Validating ${file.name}...`;
                const validExtensions = ['.pdf', '.docx', '.doc', '.txt', '.md'];
                const fileExtension = `.${file.name.split('.').pop().toLowerCase()}`;
                const MAX_DOC_SIZE = 50 * 1024 * 1024;

                if (validExtensions.includes(fileExtension) && file.size <= MAX_DOC_SIZE && file.size > 0) {
                    const item = {
                        id: Common.generateId(),
                        name: file.name,
                        size: file.size,
                        file: file,
                        status: 'Ready',
                        type: this.getFileType(file.name)
                    };
                    this.processingQueue.push(item);
                    validFilesAddedCount++;
                    lastValidFileItem = item;
                    this.addLogEntry('info', `File '${file.name}' added to processing queue.`);
                } else {
                    const reason = !validExtensions.includes(fileExtension) ? 'unsupported format' :
                        (file.size === 0 ? 'empty file' : 'file too large');
                    this.showNotification('warning', 'File Skipped', `${file.name} was skipped (${reason}).`);
                    this.addLogEntry('warning', `File '${file.name}' skipped: ${reason}.`);
                }
            }

            this.canProcess = this.processingQueue.some(item => item.status === 'Ready');
            if (validFilesAddedCount > 0) {
                this.showNotification('success', 'Files Added', `${validFilesAddedCount} file(s) added to the queue.`);
                if (lastValidFileItem && !this.activePreviewFileId) {
                    await this.setActivePreviewFile(lastValidFileItem.id);
                }
            }
            this.isLoading = false;
        },


        async addUrlToQueue() {
            const url = this.urlInput.trim();
            if (!url) {
                this.showNotification('warning', 'Invalid URL', 'Please enter a valid URL.');
                return;
            }
            if (!Common.isValidURL(url)) {
                this.showNotification('warning', 'Invalid URL', 'Please enter a valid URL.');
                return;
            }

            const item = {
                id: Common.generateId(),
                name: url,
                size: url.length,
                url: url,
                status: 'Ready',
                type: 'URL',
                isUrl: true,
                file: null
            };
            this.processingQueue.push(item);
            this.addLogEntry('info', `URL '${url}' added to processing queue.`);
            this.showNotification('success', 'URL Added', 'URL has been added to the queue.');
            this.urlInput = '';
            this.canProcess = this.processingQueue.some(i => i.status === 'Ready');
            if (!this.activePreviewFileId && this.processingQueue.length > 0) {
                await this.setActivePreviewFile(item.id);
            }
        },

        async setActivePreviewFile(itemId) {
            const fileItem = this.processingQueue.find(item => item.id === itemId);
            if (this.activePreviewFileId === itemId && this.originalContent !== '' && !this.isLoading) {
                this.addLogEntry('info', `${fileItem?.name || 'Selected item'} is already the active preview.`);
                return;
            }
            if (fileItem) {
                this.activePreviewFileId = itemId;
                this.addLogEntry('info', `Setting active preview to: ${fileItem.name}`);
                this.aiResult = null;
                this.analysisData = null;
                this.contentTab = 'original';

                if (fileItem.file) {
                    await this.displayDocumentContentOnUpload(fileItem.file);
                } else if (fileItem.isUrl) {
                    this.originalContent = `URL: ${fileItem.url}\nNo preview available. Start processing to fetch content.`;
                    this.addLogEntry('info', `Preview for URL '${fileItem.url}' is not available.`);
                } else {
                    this.originalContent = `Preview is not available for '${fileItem.name}'.`;
                    this.addLogEntry('warning', `Cannot display preview for '${fileItem.name}': No file or direct content.`);
                }
            } else {
                this.addLogEntry('warning', `Could not set active preview: item ID ${itemId} not found in queue.`);
                this.activePreviewFileId = null;
                this.originalContent = '';
                this.aiResult = null;
                this.analysisData = null;
            }
        },

        async displayDocumentContentOnUpload(fileObject) {
            if (!fileObject) {
                this.addLogEntry('warning', 'displayDocumentContentOnUpload called with no file object.');
                this.originalContent = '';
                return;
            }
            this.addLogEntry('info', `Requesting immediate text preview for ${fileObject.name}.`);
            this.isLoading = true;
            this.loadingTitle = 'Loading Preview';
            this.loadingMessage = `Extracting text from ${fileObject.name}...`;
            this.originalContent = '';

            try {
                if (this.backendConnected) {
                    const base64Data = await this.fileToBase64(fileObject);
                    const result = await eel.get_document_text_on_upload(base64Data, fileObject.name)();
                    if (result.success) {
                        this.originalContent = result.text_content;
                        this.addLogEntry('success', `Preview loaded for ${fileObject.name}.`);
                    } else {
                        throw new Error(result.error || `Failed to get preview for ${fileObject.name}.`);
                    }
                } else {
                    this.originalContent = `[DEMO PREVIEW] Content of ${fileObject.name}:\n\nThis is mock preview text.`;
                    this.addLogEntry('info', `[DEMO] Displaying mock preview for ${fileObject.name}.`);
                }
            } catch (error) {
                this.originalContent = `Error loading preview for ${fileObject.name}: ${error.message}`;
                this.addLogEntry('error', `Failed to display document content for ${fileObject.name}: ${error.message}`);
                this.showNotification('error', 'Preview Error', `Could not load preview for ${fileObject.name}.`);
            } finally {
                this.isLoading = false;
            }
        },

        async removeFromQueue(itemId) {
            const itemToRemove = this.processingQueue.find(item => item.id === itemId);
            if (itemToRemove) {
                this.addLogEntry('info', `Removing '${itemToRemove.name}' from processing queue.`);
            }
            this.processingQueue = this.processingQueue.filter(item => item.id !== itemId);
            this.canProcess = this.processingQueue.some(item => item.status === 'Ready') && !this.isProcessing;

            if (this.activePreviewFileId === itemId) {
                this.originalContent = '';
                this.activePreviewFileId = null;
                this.aiResult = null;
                this.analysisData = null;
                if (this.processingQueue.length > 0) {
                    await this.setActivePreviewFile(this.processingQueue[0].id);
                } else {
                    this.addLogEntry('info', 'Processing queue is empty. Cleared content areas.');
                }
            } else if (this.processingQueue.length === 0) {
                this.originalContent = '';
                this.activePreviewFileId = null;
                this.aiResult = null;
                this.analysisData = null;
                this.addLogEntry('info', 'Processing queue is empty. Cleared content areas.');
            }
        },

        getFileType(filename) {
            const ext = filename.toLowerCase().split('.').pop();
            const types = {
                'pdf': 'PDF',
                'docx': 'Word',
                'doc': 'Word',
                'txt': 'Text',
                'md': 'Markdown'
            };
            return types[ext] || 'File';
        },

        // 🔥 UPDATED: Main processing method với summary mode support
        async startProcessing() {
            let itemsToProcess = [];
            let currentRunIsBatch = this.processingMode === 'batch';

            if (currentRunIsBatch) {
                itemsToProcess = this.processingQueue.filter(item => item.status === 'Ready');
                if (itemsToProcess.length === 0) {
                    this.showNotification('info', 'No Items for Batch', 'No ready items found.');
                    return;
                }
            } else {
                const activeItem = this.processingQueue.find(item => item.id === this.activePreviewFileId);
                if (activeItem && activeItem.status === 'Ready') {
                    itemsToProcess.push(activeItem);
                } else {
                    const firstReadyItem = this.processingQueue.find(item => item.status === 'Ready');
                    if (firstReadyItem) {
                        itemsToProcess.push(firstReadyItem);
                        if (this.activePreviewFileId !== firstReadyItem.id) {
                            await this.setActivePreviewFile(firstReadyItem.id);
                        }
                    }
                }
            }

            if (itemsToProcess.length === 0) {
                this.showNotification('info', 'No Action', 'No items ready for processing.');
                return;
            }

            this.isProcessing = true;
            if (!currentRunIsBatch && itemsToProcess.length > 0) {
                this.aiResult = null;
                this.analysisData = null;
            }

            // 🆕 Build settings payload with summary mode
            const settingsPayload = this.buildProcessingSettings();

            // Log để debug (vì tôi paranoid về data flow)
            this.addLogEntry('debug', `Processing settings: mode=${settingsPayload.summary_mode}, ` +
                `detail=${settingsPayload.detail_level}, words=${settingsPayload.word_count_limit}`);

            try {
                if (this.backendConnected) {
                    if (currentRunIsBatch) {
                        itemsToProcess.forEach(item => item.status = 'Processing...');
                        this.addLogEntry('info', `Starting batch processing for ${itemsToProcess.length} items.`);
                        const batchPayload = {
                            files: await Promise.all(itemsToProcess.filter(i => i.file).map(async item => ({ file_data: await this.fileToBase64(item.file), filename: item.name }))),
                            urls: itemsToProcess.filter(i => i.isUrl).map(i => i.url)
                        };
                        const processId = await eel.process_document_async_web(batchPayload, settingsPayload)();
                        this.activeProcessIdToItemId[processId] = itemsToProcess.map(item => item.id);
                        this.addLogEntry('info', `Batch task (ID: ${processId}) sent to backend for ${itemsToProcess.length} items.`);
                    } else {
                        const item = itemsToProcess[0];
                        item.status = 'Processing...';
                        this.addLogEntry('info', `Starting processing for '${item.name}'.`);
                        let inputPayload;
                        if (item.isUrl) {
                            inputPayload = { url: item.url };
                        } else {
                            const base64Data = await this.fileToBase64(item.file);
                            inputPayload = { file_data: base64Data, filename: item.name };
                        }
                        const processId = await eel.process_document_async_web(inputPayload, settingsPayload)();
                        this.activeProcessIdToItemId[processId] = item.id;
                        this.addLogEntry('info', `Async processing for '${item.name}' (ID: ${processId}) sent to backend.`);
                    }
                } else {
                    this.addLogEntry('info', `[DEMO] Simulating processing. Mode: ${this.processingMode}`);
                    for (const item of itemsToProcess) {
                        item.status = 'Processing...';
                        await this.processDocumentDemo(item);
                    }
                    this.updateCanProcessAndOverallStatus();
                }
            } catch (error) {
                this.addLogEntry('error', `Failed to start processing: ${error.message}`);
                this.showNotification('error', 'Processing Start Failed', `Could not start: ${error.message}`);
                itemsToProcess.forEach(item => item.status = 'Error');
                this.updateCanProcessAndOverallStatus();
            }
        },

        // 🆕 NEW METHOD: Build processing settings với summary mode logic
        buildProcessingSettings() {
            const settings = {
                processing_mode: this.processingMode,
                language: this.targetLanguage,
                summary_mode: this.summaryMode,  // 'full' | 'percentage' | 'word-count'
            };

            // Add conditional parameters based on summary mode
            if (this.summaryMode === 'percentage') {
                settings.detail_level = this.detailLevel;
                // Backend sẽ ignore word_count_limit khi mode là percentage
            } else if (this.summaryMode === 'word-count') {
                settings.word_count_limit = this.wordCountLimit;
                // Backend sẽ ignore detail_level khi mode là word-count
            }
            // Khi summaryMode === 'full', không cần thêm params

            return settings;
        },

        async processDocumentDemo(item) {
            this.addLogEntry('info', `[DEMO] Processing: ${item.name}`);
            const currentItemIsActive = this.activePreviewFileId === item.id;
            await this.delay(1000 + Math.random() * 1500);

            let demoOriginalContent;
            if (item.file) {
                demoOriginalContent = `[DEMO CONTENT from FILE: ${item.name}]\n\nMock content.`;
            } else if (item.isUrl) {
                demoOriginalContent = `[DEMO] URL input: ${item.url}`;
            } else {
                demoOriginalContent = `[DEMO] No source for ${item.name}.`;
            }

            let demoAiResults = [];
            const langText = this.targetLanguage ? ` (lang: ${this.targetLanguage})` : '';

            // 🆕 Updated demo text với summary mode
            let summaryModeText = '';
            if (this.summaryMode === 'percentage') {
                summaryModeText = ` (${this.detailLevel}% detail)`;
            } else if (this.summaryMode === 'word-count') {
                summaryModeText = ` (max ${this.wordCountLimit} words)`;
            } else {
                summaryModeText = ' (full summary)';
            }

            demoAiResults.push({
                model: 'ChatGPT',
                content: `[DEMO] ChatGPT for ${item.name}${langText}${summaryModeText}.`,
                is_error: false
            });

            const demoAnalysisData = {
                word_count: Math.floor(Math.random() * 500) + 50,
                sentence_count: Math.floor(Math.random() * 20) + 5,
                paragraph_count: Math.floor(Math.random() * 5) + 1,
                avg_word_length: (Math.random() * 2 + 4.5).toFixed(1),
                avg_sentence_length: (Math.random() * 5 + 10).toFixed(1),
                common_words: [['demo', 5], ['text', 4], ['content', 3]],
                error: null
            };

            const demoResultDict = {
                success: true,
                original_text: demoOriginalContent,
                ai_result: demoAiResults[0],
                analysis: demoAnalysisData,
                error: null,
                has_errors: demoAiResults.some(r => r.is_error)
            };

            this.handleProcessingResult(demoResultDict, item);
            item.status = 'Completed';
            this.stats.processed++;

            if (currentItemIsActive) {
                this.originalContent = demoResultDict.original_text;
                this.aiResult = demoResultDict.ai_result;
                this.analysisData = demoResultDict.analysis;
                if (this.analysisData && this.analysisData.common_words && this.contentTab === 'analysis') {
                    this.$nextTick(() => this.renderWordCloud(this.analysisData.common_words));
                }
            }
            this.addLogEntry('info', `[DEMO] Done: ${item.name}.`);
        },

        handleProcessingResult(result, item) {
            if (result && typeof result.success !== 'undefined') {
                const completionMessage = `Processing of '${item.name}' concluded.`;
                this.addLogEntry(result.success && !result.has_errors ? 'success' : 'warning',
                    completionMessage + (result.error ? ` Issues: ${result.error}` : ''));

                if (this.activePreviewFileId === item.id) {
                    if (item.isUrl) {
                        const urlLink = `<a href="${item.url}" target="_blank" class="text-blue-600 underline">${item.url}</a>`;
                        this.originalContent = result.original_text ? `${urlLink}<br>${result.original_text}` : urlLink;
                    } else {
                        this.originalContent = result.original_text || `Content for ${item.name} unavailable.`;
                    }
                    this.aiResult = result.ai_result || null;
                    this.analysisData = result.analysis || null;
                    if (this.analysisData && this.analysisData.common_words && this.contentTab === 'analysis' && !this.analysisData.error) {
                        this.$nextTick(() => this.renderWordCloud(this.analysisData.common_words));
                    } else if (this.contentTab === 'analysis' && window.ChartRenderer) {
                        ChartRenderer.destroyChart('wordFrequencyChart');
                    }
                }

                if (result.error && result.error.trim() !== "") {
                    this.showNotification(result.success && !result.has_errors ? 'warning' : 'error',
                        'Processing Note', `${item.name}: ${result.error}`);
                } else if (result.success && !result.has_errors && this.processingMode === 'individual') {
                    this.showNotification('success', 'Processing Complete', `${item.name} processed.`);
                } else if (result.success === false || result.has_errors) {
                    this.showNotification('error', 'Processing Failed', `${item.name} not processed fully.`);
                }
            } else {
                const errorDetail = `Processing for ${item.name} returned invalid result.`;
                this.addLogEntry('error', errorDetail);
                this.showNotification('error', 'Backend Error', errorDetail);
                if (this.activePreviewFileId === item.id) {
                    this.originalContent = `Failed to process ${item.name}.`;
                    this.aiResult = null;
                    this.analysisData = null;
                }
            }
        },

        handleEelDocProcessingProgress(processId, percent, message) {
            this.addLogEntry('info', `[Doc PID: ${processId}] ${percent}% - ${message}`);
            const itemOrItemIds = this.activeProcessIdToItemId[processId];
            if (itemOrItemIds && this.isProcessing) {
                this.loadingTitle = `Processing (${percent}%)`;
                this.loadingMessage = message;
            }
        },

        handleEelProcessingComplete(processId, resultDict) {
            const itemOrItemIds = this.activeProcessIdToItemId[processId];
            if (!itemOrItemIds) {
                this.addLogEntry('warning', `processingComplete for unknown PID: ${processId}`);
                this.updateCanProcessAndOverallStatus();
                return;
            }

            const processSingleItemResult = (itemId, resDict) => {
                const item = this.processingQueue.find(i => i.id === itemId);
                if (item) {
                    console.log("Received resultDict for item " + item.name + ":", JSON.parse(JSON.stringify(resDict)));
                    if (resDict.success) {
                        item.status = 'Completed';
                        if (resDict.has_errors) {
                            this.addLogEntry('warning', `Item '${item.name}' (ID: ${item.id}) completed successfully, but with partial errors noted by backend (e.g. AI model issues). Check AI results details.`);
                        }
                    } else {
                        item.status = 'Error';
                    }
                    this.handleProcessingResult(resDict, item);
                    if (item.status === 'Completed') {
                        this.stats.processed++;
                    }
                } else {
                    this.addLogEntry('warning', `Item ID ${itemId} (PID ${processId}) not found.`);
                }
            };

            if (Array.isArray(itemOrItemIds)) {
                this.addLogEntry('info', `Batch (PID: ${processId}) completed.`);
                console.log("Batch result received:", JSON.parse(JSON.stringify(resultDict)));

                itemOrItemIds.forEach(itemId => {
                    const item = this.processingQueue.find(i => i.id === itemId);
                    if (item) {
                        console.log(`Checking item: ${item.name}`);
                        console.log(`Processed files:`, resultDict.processed_files);
                        console.log(`Failed files:`, resultDict.failed_files);

                        const wasProcessed = resultDict.processed_files?.includes(item.name);
                        const hadFailure = resultDict.failed_files?.some(f => f[0] === item.name);

                        console.log(`Item ${item.name}: wasProcessed=${wasProcessed}, hadFailure=${hadFailure}`);

                        if (resultDict.success && !hadFailure) {
                            item.status = 'Completed';
                            this.stats.processed++;
                            console.log(`Item ${item.name} marked as Completed`);
                        } else {
                            item.status = 'Error';
                            console.log(`Item ${item.name} marked as Error - success: ${resultDict.success}, hadFailure: ${hadFailure}`);
                        }
                    }
                });

                this.originalContent = `Batch (ID: ${processId}) Results:\nProcessed: ${resultDict.processed_files?.join(', ') || 'None'}\nFailed: ${resultDict.failed_files?.map(f => f[0]).join(', ') || 'None'}\nConcatenated Chars: ${resultDict.concatenated_text_char_count || 0}`;
                this.aiResult = resultDict.ai_result || null;
                this.analysisData = null;
                this.showNotification(resultDict.success && !resultDict.has_errors ? 'success' : 'warning',
                    'Batch Process Complete', `Task (ID: ${processId}) finished. ${resultDict.error || ''}`);
                if (this.aiResult) this.contentTab = 'results';
            } else {
                processSingleItemResult(itemOrItemIds, resultDict);
            }

            delete this.activeProcessIdToItemId[processId];
            this.updateCanProcessAndOverallStatus();
        },

        handleEelProcessingError(processId, errorMessageStr) {
            const itemOrItemIds = this.activeProcessIdToItemId[processId];
            if (!itemOrItemIds) {
                this.addLogEntry('warning', `processingError for unknown PID: ${processId}`);
                this.updateCanProcessAndOverallStatus();
                return;
            }

            const handleErrorForItem = (itemId, errMsg) => {
                const item = this.processingQueue.find(i => i.id === itemId);
                if (item) {
                    item.status = 'Error';
                    this.showNotification('error', `Processing Error: ${item.name}`, errMsg);
                    this.addLogEntry('error', `Processing Error for '${item.name}' (Eel ID: ${processId}): ${errMsg}`);
                    if (this.activePreviewFileId === item.id) {
                        this.originalContent = `Error processing ${item.name}: ${errMsg}`;
                        this.aiResult = { model: 'Error', content: `Processing failed: ${errMsg}`, is_error: true };
                        this.analysisData = null;
                    }
                }
            };

            if (Array.isArray(itemOrItemIds)) {
                this.addLogEntry('error', `Batch task (ID: ${processId}) error: ${errorMessageStr}`);
                this.showNotification('error', 'Batch Processing Error', `Task ID ${processId}: ${errorMessageStr}`);
                itemOrItemIds.forEach(id => handleErrorForItem(id, "Part of failed batch."));
            } else {
                handleErrorForItem(itemOrItemIds, errorMessageStr);
            }

            delete this.activeProcessIdToItemId[processId];
            this.updateCanProcessAndOverallStatus();
        },

        updateCanProcessAndOverallStatus() {
            const stillProcessingInQueue = this.processingQueue.some(i => i.status === 'Processing...');
            this.isProcessing = stillProcessingInQueue;
            this.canProcess = this.processingQueue.some(item => item.status === 'Ready') && !stillProcessingInQueue;
            if (!stillProcessingInQueue) {
                this.addLogEntry('info', 'All current processing tasks concluded.');
                this.loadingTitle = '';
                this.loadingMessage = '';
            }
        },

        // ===== UTILITY METHODS (không đổi) =====
        formatFileSize(bytes) {
            if (bytes == null || isNaN(bytes) || bytes < 0) return 'N/A';
            if (bytes === 0) return '0 B';
            const k = 1024;
            const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
        },

        get filteredResults() {
            if (!this.searchResults || this.searchResults.length === 0) return [];
            if (this.resultsFilter === 'all') return this.searchResults;
            if (this.resultsFilter === 'social') return this.searchResults.filter(r => r.is_social_media);
            return this.searchResults.filter(result => result.is_social_media && this.getSocialPlatform(result.link) === this.resultsFilter);
        },

        getSocialPlatform(url) {
            if (!url) return 'unknown';
            const urlLower = url.toLowerCase();
            const platforms = {
                'facebook.com': 'facebook', 'fb.com': 'facebook', 'instagram.com': 'instagram',
                'twitter.com': 'twitter', 'x.com': 'twitter', 'linkedin.com': 'linkedin',
                'tiktok.com': 'tiktok', 'youtube.com': 'youtube',
                'pinterest.com': 'pinterest', 'reddit.com': 'reddit'
            };
            for (const [domain, platform] of Object.entries(platforms)) {
                if (urlLower.includes(domain)) return platform;
            }
            return 'other_social';
        },

        getAIModelIcon(modelName) {
            const icons = {
                'DeepSeek': 'fas fa-brain',
                'Grok': 'fas fa-infinity',
                'ChatGPT': 'fas fa-bolt',
                'N/A': 'fas fa-ban',
                'Error': 'fas fa-exclamation-triangle'
            };
            return icons[modelName] || 'fas fa-microchip';
        },

        openResult(url) {
            if (url) {
                window.open(url, '_blank');
                this.addLogEntry('info', `Opened result link: ${url}`);
            }
        },

        showNotification(type, title, message) {
            if (window.Notifications && typeof window.Notifications.show === 'function') {
                window.Notifications.show(type, title, message);
            } else {
                this.notification = { show: true, type, title, message };
                setTimeout(() => { this.notification.show = false; }, 5000);
            }
        },

        addLogEntry(level, message) {
            const entry = {
                id: Common.generateId ? Common.generateId() : Date.now() + Math.random(),
                timestamp: new Date().toLocaleTimeString([], {
                    hour: '2-digit',
                    minute: '2-digit',
                    second: '2-digit',
                    hour12: false
                }),
                level: level.toUpperCase(),
                message
            };
            this.logEntries.unshift(entry);
            if (this.logEntries.length > 200) this.logEntries = this.logEntries.slice(0, 200);
            const consoleMethod = level === 'error' ? console.error : (level === 'warning' ? console.warn : console.log);
            consoleMethod(`[${entry.timestamp} ${entry.level}] ${entry.message}`);
        },

        async fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = (error) => reject(new Error(`FileReader error: ${error.message || error}`));
                reader.readAsDataURL(file);
            });
        },

        delay(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        },

        getMockSearchResults() {
            return [
                { title: "Summer Demo", link: "https://example.com/f", displayed_link: "fb.com", snippet: "Beach! #summer", is_social_media: true, source: "ImgRes" },
            ];
        },

        highlightSearchTerm(content, term) {
            if (!term || term.trim() === '' || !content) return content;
            try {
                const escapedTerm = term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const regex = new RegExp(`(${escapedTerm})`, 'gi');
                const stringContent = String(content);
                return stringContent.replace(regex, '<mark class="bg-yellow-200 rounded px-0.5 selection:bg-blue-200 selection:text-white">$1</mark>');
            } catch (e) {
                console.warn("Error in highlightSearchTerm:", e);
                return content;
            }
        },

        renderWordCloud(commonWords) {
            this.$nextTick(() => {
                if (window.ChartRenderer && this.contentTab === 'analysis' && commonWords && commonWords.length > 0 && this.analysisData && !this.analysisData.error) {
                    try {
                        ChartRenderer.createWordFrequencyChart('wordFrequencyChart', commonWords);
                        this.addLogEntry('info', 'Word frequency chart rendered.');
                    } catch (e) {
                        this.addLogEntry('error', `Failed to render word cloud: ${e.message}`);
                        console.error("Chart rendering error:", e);
                    }
                } else if (this.contentTab === 'analysis' && window.ChartRenderer) {
                    ChartRenderer.destroyChart('wordFrequencyChart');
                }
            });
        },

        async performFullCleanup() {
            this.isLoading = true;
            this.loadingTitle = "System Cleanup";
            this.loadingMessage = "Requesting cleanup of temporary files...";
            this.addLogEntry('info', "User initiated temporary file cleanup.");
            try {
                if (typeof eel !== 'undefined') {
                    const result = await eel.perform_cleanup_temp_files()();
                    if (result.success) {
                        this.showNotification('success', 'Cleanup Successful', `Cleaned ${result.cleaned_count} temporary files.${result.errors?.length ? ' Some errors occurred.' : ''}`);
                        this.addLogEntry('success', `Temp file cleanup: ${result.cleaned_count} deleted. Errors: ${result.errors?.join(', ') || 'None'}`);
                    } else {
                        throw new Error(result.error || "Cleanup failed on backend.");
                    }
                } else {
                    await this.delay(1000);
                    this.showNotification('info', 'Demo Mode', 'Cleanup function called (no actual files deleted).');
                    this.addLogEntry('info', "[DEMO] Temporary file cleanup simulated.");
                }
            } catch (error) {
                this.addLogEntry('error', `Cleanup failed: ${error.message}`);
                this.showNotification('error', 'Cleanup Failed', error.message);
            } finally {
                this.isLoading = false;
            }
        },

        async copyContentToClipboard() {
            let contentToCopy = '';
            let contentType = '';
            if (this.contentTab === 'original') {
                contentToCopy = this.originalContent;
                contentType = 'Original Content';
            } else if (this.contentTab === 'results') {
                contentType = 'AI Result';
                const result = this.aiResult;
                if (result && !result.is_error && result.content) {
                    contentToCopy = result.content;
                } else if (result && result.is_error && result.content) {
                    contentToCopy = `ERROR: ${result.content}`;
                }
            } else if (this.contentTab === 'analysis' && this.analysisData) {
                contentType = 'Analysis Data';
                if (this.analysisData.error) {
                    contentToCopy = `Analysis Error: ${this.analysisData.error}\n`;
                } else {
                    contentToCopy += `Word Count: ${this.analysisData.word_count?.toLocaleString() || 'N/A'}\nSentence Count: ${this.analysisData.sentence_count?.toLocaleString() || 'N/A'}\nParagraph Count: ${this.analysisData.paragraph_count?.toLocaleString() || 'N/A'}\nAvg. Word Length: ${this.analysisData.avg_word_length ? Number(this.analysisData.avg_word_length).toFixed(1) : 'N/A'}\nAvg. Sentence Length: ${this.analysisData.avg_sentence_length ? Number(this.analysisData.avg_sentence_length).toFixed(1) + ' words' : 'N/A'}\n\n`;
                    if (this.analysisData.common_words && this.analysisData.common_words.length > 0) {
                        contentToCopy += "Common Words:\n";
                        this.analysisData.common_words.slice(0, 15).forEach(pair => contentToCopy += `- ${pair[0]} (${pair[1]})\n`);
                    }
                }
            }

            if (contentToCopy && contentToCopy.trim() !== '') {
                try {
                    await navigator.clipboard.writeText(contentToCopy.trim());
                    this.showNotification('success', 'Copied to Clipboard', `${contentType} copied.`);
                    this.addLogEntry('info', `${contentType} copied to clipboard.`);
                } catch (err) {
                    this.showNotification('error', 'Copy Failed', 'Could not copy.');
                    this.addLogEntry('error', `Failed to copy ${contentType}: ${err}`);
                }
            } else {
                this.showNotification('info', 'Nothing to Copy', `No content in '${this.contentTab}' tab.`);
                this.addLogEntry('info', `Copy attempt: No content in '${this.contentTab}' tab.`);
            }
        }
    };
}

// Edge case protection: Ensure Common.generateId exists
if (typeof Common === 'undefined' || typeof Common.generateId === 'undefined') {
    console.warn('Common.js or Common.generateId not loaded/defined before app.js. Using fallback for ID generation.');
    window.Common = window.Common || {};
    window.Common.generateId = () => `id_${Date.now().toString(36)}_${Math.random().toString(36).substring(2, 9)}`;
}