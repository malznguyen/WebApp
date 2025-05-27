// Enhanced Toolkit v2.0 - Main App Logic
function app() {
    return {
        // ===== STATE MANAGEMENT =====
        activeTab: 'image-search',
        statusMessage: 'Ready',
        isLoading: false,
        loadingTitle: '',
        loadingMessage: '',

        // ===== IMAGE SEARCH STATE =====
        searchOptions: {
            socialOnly: false
        },
        isSearching: false,
        searchProgress: 0,
        searchStatus: '',
        searchResults: [],
        resultsFilter: 'all',
        viewMode: 'list',
        canSearch: false,
        selectedImage: null,

        // ===== DOCUMENT PROCESSING STATE =====
        processingMode: 'individual',
        aiModels: {
            deepseek: true,
            grok: false,
            chatgpt: false
        },
        detailLevel: 50,
        targetLanguage: null,
        isProcessing: false,
        processingQueue: [],
        originalContent: '',
        aiResults: [],
        analysisData: null,
        contentTab: 'original',
        canProcess: false,
        settingsPanels: {
            processing: true,
            aiModels: true,
            advanced: false
        },

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
            console.log('🚀 Enhanced Toolkit v2.0 - Initializing...');
            
            try {
                await this.loadAppConfig();
                this.setupEventListeners();
                this.addLogEntry('info', 'Application initialized successfully');
                this.statusMessage = 'Ready';
            } catch (error) {
                console.error('Initialization error:', error);
                this.showNotification('error', 'Initialization Error', 'Failed to initialize application');
                this.addLogEntry('error', `Initialization failed: ${error.message}`);
            }
        },

        async loadAppConfig() {
            try {
                if (typeof eel !== 'undefined') {
                    const config = await eel.get_app_config()();
                    this.addLogEntry('info', `API Configuration loaded - SERP: ${config.has_serp_api}, Imgur: ${config.has_imgur}`);
                    return config;
                }
            } catch (error) {
                this.addLogEntry('warning', 'Running in demo mode - Python backend not connected');
                return this.getMockConfig();
            }
        },

        getMockConfig() {
            return {
                has_serp_api: true,
                has_imgur: true,
                has_deepseek: true,
                has_grok: false,
                has_chatgpt: true,
                supported_formats: ['.pdf', '.docx', '.txt', '.md']
            };
        },

        setupEventListeners() {
            // File drop handlers
            document.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.stopPropagation();
            });

            document.addEventListener('drop', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                if (this.activeTab === 'image-search') {
                    this.handleImageDrop(e);
                } else if (this.activeTab === 'document-summary') {
                    this.handleDocumentDrop(e);
                }
            });

            // Keyboard shortcuts
            document.addEventListener('keydown', (e) => {
                if (e.ctrlKey || e.metaKey) {
                    switch (e.key) {
                        case '1':
                            e.preventDefault();
                            this.activeTab = 'image-search';
                            break;
                        case '2':
                            e.preventDefault();
                            this.activeTab = 'document-summary';
                            break;
                        case 'l':
                            e.preventDefault();
                            this.logPanelExpanded = !this.logPanelExpanded;
                            break;
                    }
                }
            });
        },

        // ===== IMAGE SEARCH METHODS =====
        handleImageDrop(event) {
            const files = Array.from(event.dataTransfer.files);
            const imageFiles = files.filter(file => file.type.startsWith('image/'));
            
            if (imageFiles.length > 0) {
                this.processImageFile(imageFiles[0]);
            } else {
                this.showNotification('error', 'Invalid File', 'Please drop an image file');
            }
        },

        async processImageFile(file) {
            this.addLogEntry('info', `Processing image: ${file.name} (${(file.size / 1024 / 1024).toFixed(2)}MB)`);
            
            try {
                // Validate file
                if (file.size > 10 * 1024 * 1024) { // 10MB limit
                    throw new Error('File too large (max 10MB)');
                }

                // Convert to base64
                const base64 = await this.fileToBase64(file);
                this.selectedImage = {
                    name: file.name,
                    size: file.size,
                    data: base64,
                    preview: base64 // For image preview
                };
                
                this.canSearch = true;
                console.log('✅ Image processed successfully, canSearch:', this.canSearch);
                this.showNotification('success', 'Image Loaded', 'Ready to search');
                
            } catch (error) {
                this.addLogEntry('error', `Image processing failed: ${error.message}`);
                this.showNotification('error', 'Processing Error', error.message);
                this.canSearch = false;
            }
        },

        browseImageFiles() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = 'image/*';
            input.onchange = (e) => {
                if (e.target.files.length > 0) {
                    this.processImageFile(e.target.files[0]);
                }
            };
            input.click();
        },

        removeSelectedImage() {
            this.selectedImage = null;
            this.canSearch = false;
            this.searchResults = [];
            this.addLogEntry('info', 'Image removed from search');
        },

        async startImageSearch() {
            if (!this.selectedImage || this.isSearching) return;

            this.isSearching = true;
            this.searchProgress = 0;
            this.searchResults = [];
            this.searchStatus = 'Initializing search...';
            
            try {
                this.addLogEntry('info', `Starting image search - Social only: ${this.searchOptions.socialOnly}`);
                
                // Update progress
                this.updateSearchProgress(10, 'Uploading image...');
                
                if (typeof eel !== 'undefined') {
                    // Real backend call
                    const result = await eel.search_image_web(
                        this.selectedImage.data,
                        this.selectedImage.name,
                        this.searchOptions.socialOnly
                    )();
                    
                    this.handleSearchResults(result);
                } else {
                    // Demo mode
                    await this.runDemoSearch();
                }
                
            } catch (error) {
                this.addLogEntry('error', `Search failed: ${error.message}`);
                this.showNotification('error', 'Search Failed', error.message);
            } finally {
                this.isSearching = false;
                this.searchProgress = 100;
                this.searchStatus = 'Search completed';
            }
        },

        async runDemoSearch() {
            // Demo search with fake progress
            const steps = [
                { progress: 20, message: 'Uploading to Imgur...' },
                { progress: 40, message: 'Analyzing image...' },
                { progress: 60, message: 'Querying search engines...' },
                { progress: 80, message: 'Processing results...' },
                { progress: 100, message: 'Search completed' }
            ];

            for (const step of steps) {
                await this.delay(800);
                this.updateSearchProgress(step.progress, step.message);
            }

            // Mock results
            this.searchResults = this.getMockSearchResults();
            this.addLogEntry('info', `Search completed - Found ${this.searchResults.length} results`);
        },

        handleSearchResults(result) {
            if (result.success) {
                this.searchResults = result.results || [];
                this.addLogEntry('info', `Search completed - Found ${this.searchResults.length} results`);
                this.showNotification('success', 'Search Complete', `Found ${this.searchResults.length} results`);
            } else {
                throw new Error(result.error || 'Search failed');
            }
        },

        updateSearchProgress(progress, message) {
            this.searchProgress = progress;
            this.searchStatus = message;
        },

        // ===== DOCUMENT PROCESSING METHODS =====
        handleDocumentDrop(event) {
            const files = Array.from(event.dataTransfer.files);
            const validFiles = files.filter(file => 
                ['.pdf', '.docx', '.txt', '.md'].some(ext => 
                    file.name.toLowerCase().endsWith(ext)
                )
            );
            
            if (validFiles.length > 0) {
                this.addFilesToQueue(validFiles);
            } else {
                this.showNotification('error', 'Invalid Files', 'Please drop PDF, DOCX, TXT, or MD files');
            }
        },

        browseDocumentFiles() {
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.pdf,.docx,.txt,.md';
            input.multiple = true; // Allow multiple file selection
            input.onchange = (e) => {
                if (e.target.files.length > 0) {
                    this.addFilesToQueue(Array.from(e.target.files));
                }
            };
            input.click();
        },

        addFilesToQueue(files) {
            const validFiles = files.filter(file => {
                const validExtensions = ['.pdf', '.docx', '.txt', '.md'];
                return validExtensions.some(ext => 
                    file.name.toLowerCase().endsWith(ext)
                );
            });

            if (validFiles.length === 0) {
                this.showNotification('error', 'Invalid Files', 'Please select PDF, DOCX, TXT, or MD files');
                return;
            }

            validFiles.forEach(file => {
                const item = {
                    id: Date.now() + Math.random(),
                    name: file.name,
                    size: file.size,
                    file: file,
                    status: 'Ready',
                    type: this.getFileType(file.name)
                };
                this.processingQueue.push(item);
            });
            
            this.canProcess = this.processingQueue.length > 0;
            this.addLogEntry('info', `Added ${validFiles.length} files to processing queue`);
            
            if (files.length > validFiles.length) {
                const skipped = files.length - validFiles.length;
                this.showNotification('warning', 'Some Files Skipped', 
                    `${skipped} files were skipped (unsupported format)`);
            }
        },

        removeFromQueue(itemId) {
            this.processingQueue = this.processingQueue.filter(item => item.id !== itemId);
            this.canProcess = this.processingQueue.length > 0;
            this.addLogEntry('info', 'File removed from processing queue');
        },

        getFileType(filename) {
            const ext = filename.toLowerCase().split('.').pop();
            const types = {
                'pdf': 'PDF Document',
                'docx': 'Word Document', 
                'txt': 'Text File',
                'md': 'Markdown File'
            };
            return types[ext] || 'Unknown';
        },

        async startProcessing() {
            if (!this.canProcess || this.isProcessing) return;

            this.isProcessing = true;
            this.aiResults = [];
            this.analysisData = null;
            
            try {
                this.addLogEntry('info', `Starting document processing - Mode: ${this.processingMode}`);
                
                for (const item of this.processingQueue) {
                    item.status = 'Processing...';
                    
                    if (typeof eel !== 'undefined') {
                        // Real backend processing
                        const result = await this.processDocumentReal(item);
                        this.handleProcessingResult(result, item);
                    } else {
                        // Demo processing
                        await this.processDocumentDemo(item);
                    }
                    
                    item.status = 'Completed';
                }
                
            } catch (error) {
                this.addLogEntry('error', `Processing failed: ${error.message}`);
                this.showNotification('error', 'Processing Failed', error.message);
            } finally {
                this.isProcessing = false;
            }
        },

        async processDocumentReal(item) {
            try {
                // Convert file to base64 for upload
                const base64Data = await this.fileToBase64(item.file);
                
                // Upload file first
                const uploadResult = await eel.upload_file_web(base64Data, item.name)();
                if (!uploadResult.success) {
                    throw new Error(uploadResult.error);
                }
                
                // Process document
                const settings = {
                    processing_mode: this.processingMode,
                    detail_level: this.detailLevel,
                    ai_models: this.aiModels,
                    language: this.targetLanguage || null
                };
                
                const result = await eel.process_document_web(uploadResult.temp_path, settings)();
                return result;
                
            } catch (error) {
                this.addLogEntry('error', `Document processing failed: ${error.message}`);
                throw error;
            }
        },

        async processDocumentDemo(item) {
            await this.delay(2000);
            
            // Mock processing result
            this.originalContent = `Sample content from ${item.name}\n\nThis is a demonstration of the document processing feature. In the real application, this would contain the extracted text from your document.`;
            
            this.aiResults = [];
            
            if (this.aiModels.deepseek) {
                this.aiResults.push({
                    model: 'DeepSeek',
                    content: 'This is a sample AI-generated summary from DeepSeek model. The actual summary would analyze your document content and provide intelligent insights.'
                });
            }
            
            if (this.aiModels.grok) {
                this.aiResults.push({
                    model: 'Grok',
                    content: 'This is a sample AI-generated summary from Grok model. It would provide a different perspective and analysis of your document.'
                });
            }
            
            if (this.aiModels.chatgpt) {
                this.aiResults.push({
                    model: 'ChatGPT',
                    content: 'This is a sample AI-generated summary from ChatGPT model. It would provide comprehensive analysis and insights from your document.'
                });
            }
            
            this.analysisData = {
                word_count: 1247,
                sentence_count: 89,
                paragraph_count: 12,
                avg_word_length: 4.2,
                avg_sentence_length: 14.0
            };
        },

        handleProcessingResult(result, item) {
            if (result.success) {
                this.originalContent = result.original_text || '';
                this.aiResults = result.ai_results ? result.ai_results.filter(r => r !== null) : [];
                this.analysisData = result.analysis || null;
                this.addLogEntry('info', `Processing completed for: ${item.name}`);
            } else {
                throw new Error(result.error || 'Processing failed');
            }
        },

        // ===== UTILITY METHODS =====
        get filteredResults() {
            if (this.resultsFilter === 'all') return this.searchResults;
            if (this.resultsFilter === 'social') return this.searchResults.filter(r => r.is_social_media);
            
            return this.searchResults.filter(result => 
                this.getSocialPlatform(result.link) === this.resultsFilter
            );
        },

        getSocialPlatform(url) {
            const platforms = {
                'facebook.com': 'facebook',
                'instagram.com': 'instagram', 
                'twitter.com': 'twitter',
                'x.com': 'twitter',
                'linkedin.com': 'linkedin',
                'tiktok.com': 'tiktok'
            };
            
            for (const [domain, platform] of Object.entries(platforms)) {
                if (url.toLowerCase().includes(domain)) {
                    return platform;
                }
            }
            return 'social';
        },

        openResult(url) {
            window.open(url, '_blank');
            this.addLogEntry('info', `Opened result: ${url}`);
        },

        showNotification(type, title, message) {
            this.notification = {
                show: true,
                type,
                title,
                message
            };
            
            setTimeout(() => {
                this.notification.show = false;
            }, 5000);
        },

        addLogEntry(level, message) {
            const entry = {
                id: Date.now() + Math.random(),
                timestamp: new Date().toLocaleTimeString(),
                level: level.toUpperCase(),
                message
            };
            
            this.logEntries.unshift(entry);
            
            // Keep only last 100 entries
            if (this.logEntries.length > 100) {
                this.logEntries = this.logEntries.slice(0, 100);
            }
            
            console.log(`[${entry.level}] ${entry.message}`);
        },

        async fileToBase64(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => resolve(reader.result);
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        },

        delay(ms) {
            return new Promise(resolve => setTimeout(resolve, ms));
        },

        getMockSearchResults() {
            return [
                {
                    title: "Sample Social Media Post",
                    link: "https://facebook.com/sample",
                    displayed_link: "facebook.com",
                    snippet: "This is a sample social media result that would appear in real search results.",
                    is_social_media: true,
                    source: "Images"
                },
                {
                    title: "Sample Instagram Post", 
                    link: "https://instagram.com/sample",
                    displayed_link: "instagram.com", 
                    snippet: "Another sample result from Instagram platform.",
                    is_social_media: true,
                    source: "Images"
                },
                {
                    title: "Regular Web Result",
                    link: "https://example.com/sample",
                    displayed_link: "example.com",
                    snippet: "This is a sample regular web result that would appear in search.",
                    is_social_media: false,
                    source: "Organic Search"
                }
            ];
        }
    };
}