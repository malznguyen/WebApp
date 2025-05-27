// Enhanced Toolkit v2.0 - File Handler Utility
// Advanced file handling, drag/drop, and processing utilities

const FileHandler = {
    // File type configurations
    fileTypes: {
        image: {
            extensions: ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'],
            mimeTypes: ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/bmp', 'image/svg+xml'],
            maxSize: 10 * 1024 * 1024, // 10MB
            description: 'Image files'
        },
        document: {
            extensions: ['.pdf', '.docx', '.txt', '.md'],
            mimeTypes: ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain', 'text/markdown'],
            maxSize: 50 * 1024 * 1024, // 50MB
            description: 'Document files'
        }
    },

    // Initialize drag and drop for an element
    initializeDragDrop(element, options = {}) {
        const {
            onDragEnter = () => {},
            onDragLeave = () => {},
            onDragOver = () => {},
            onDrop = () => {},
            allowedTypes = 'all',
            multiple = false,
            previewContainer = null
        } = options;

        let dragCounter = 0;

        // Prevent default browser behavior
        const preventDefaults = (e) => {
            e.preventDefault();
            e.stopPropagation();
        };

        // Handle drag enter
        const handleDragEnter = (e) => {
            preventDefaults(e);
            dragCounter++;
            element.classList.add('drag-active');
            onDragEnter(e);
        };

        // Handle drag leave
        const handleDragLeave = (e) => {
            preventDefaults(e);
            dragCounter--;
            if (dragCounter === 0) {
                element.classList.remove('drag-active');
                onDragLeave(e);
            }
        };

        // Handle drag over
        const handleDragOver = (e) => {
            preventDefaults(e);
            onDragOver(e);
        };

        // Handle drop
        const handleDrop = (e) => {
            preventDefaults(e);
            dragCounter = 0;
            element.classList.remove('drag-active');
            
            const files = Array.from(e.dataTransfer.files);
            const validFiles = this.filterFilesByType(files, allowedTypes);
            
            if (validFiles.length > 0) {
                const filesToProcess = multiple ? validFiles : [validFiles[0]];
                onDrop(filesToProcess, e);
            }
        };

        // Add event listeners
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            element.addEventListener(eventName, preventDefaults, false);
        });

        element.addEventListener('dragenter', handleDragEnter, false);
        element.addEventListener('dragleave', handleDragLeave, false);
        element.addEventListener('dragover', handleDragOver, false);
        element.addEventListener('drop', handleDrop, false);

        // Return cleanup function
        return () => {
            element.removeEventListener('dragenter', handleDragEnter);
            element.removeEventListener('dragleave', handleDragLeave);
            element.removeEventListener('dragover', handleDragOver);
            element.removeEventListener('drop', handleDrop);
        };
    },

    // Filter files by type
    filterFilesByType(files, allowedTypes) {
        if (allowedTypes === 'all') return files;
        
        const typeConfig = this.fileTypes[allowedTypes];
        if (!typeConfig) return files;

        return files.filter(file => {
            const extension = '.' + file.name.split('.').pop().toLowerCase();
            return typeConfig.extensions.includes(extension) || 
                   typeConfig.mimeTypes.includes(file.type);
        });
    },

    // Validate file
    validateFile(file, allowedTypes = 'all') {
        const errors = [];

        // Check if file type is allowed
        if (allowedTypes !== 'all') {
            const typeConfig = this.fileTypes[allowedTypes];
            if (typeConfig) {
                const extension = '.' + file.name.split('.').pop().toLowerCase();
                const isValidExtension = typeConfig.extensions.includes(extension);
                const isValidMimeType = typeConfig.mimeTypes.includes(file.type);
                
                if (!isValidExtension && !isValidMimeType) {
                    errors.push(`Invalid file type. Allowed: ${typeConfig.extensions.join(', ')}`);
                }

                // Check file size
                if (file.size > typeConfig.maxSize) {
                    const maxSizeMB = typeConfig.maxSize / (1024 * 1024);
                    errors.push(`File too large. Maximum size: ${maxSizeMB}MB`);
                }
            }
        }

        // Check for empty files
        if (file.size === 0) {
            errors.push('File is empty');
        }

        // Check for suspicious file names
        if (this.isSuspiciousFileName(file.name)) {
            errors.push('Suspicious file name detected');
        }

        return {
            valid: errors.length === 0,
            errors
        };
    },

    // Check for suspicious file names
    isSuspiciousFileName(filename) {
        const suspiciousPatterns = [
            /^\./, // Hidden files
            /\.(exe|bat|cmd|scr|com|pif|vbs|js|jar)$/i, // Executable files
            /[<>:"|?*]/, // Invalid characters
            /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(\.|$)/i // Reserved names
        ];

        return suspiciousPatterns.some(pattern => pattern.test(filename));
    },

    // Read file as different formats
    readFile(file, format = 'text') {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = (e) => reject(new Error('Failed to read file'));

            switch (format) {
                case 'text':
                    reader.readAsText(file);
                    break;
                case 'dataUrl':
                    reader.readAsDataURL(file);
                    break;
                case 'arrayBuffer':
                    reader.readAsArrayBuffer(file);
                    break;
                case 'binaryString':
                    reader.readAsBinaryString(file);
                    break;
                default:
                    reject(new Error('Invalid format specified'));
            }
        });
    },

    // Read multiple files
    async readMultipleFiles(files, format = 'text', progressCallback = null) {
        const results = [];
        const total = files.length;

        for (let i = 0; i < files.length; i++) {
            try {
                const content = await this.readFile(files[i], format);
                results.push({
                    file: files[i],
                    content,
                    success: true
                });
            } catch (error) {
                results.push({
                    file: files[i],
                    error: error.message,
                    success: false
                });
            }

            if (progressCallback) {
                progressCallback((i + 1) / total * 100, `Processed ${i + 1}/${total} files`);
            }
        }

        return results;
    },

    // Create file preview
    async createFilePreview(file, previewType = 'auto') {
        const fileType = this.getFileType(file);
        
        try {
            switch (previewType === 'auto' ? fileType : previewType) {
                case 'image':
                    return await this.createImagePreview(file);
                case 'text':
                    return await this.createTextPreview(file);
                case 'document':
                    return await this.createDocumentPreview(file);
                default:
                    return this.createGenericPreview(file);
            }
        } catch (error) {
            console.warn('Preview creation failed:', error);
            return this.createGenericPreview(file);
        }
    },

    // Create image preview
    async createImagePreview(file) {
        const dataUrl = await this.readFile(file, 'dataUrl');
        return {
            type: 'image',
            content: dataUrl,
            metadata: await this.getImageMetadata(file)
        };
    },

    // Create text preview
    async createTextPreview(file, maxLength = 500) {
        const content = await this.readFile(file, 'text');
        return {
            type: 'text',
            content: content.slice(0, maxLength) + (content.length > maxLength ? '...' : ''),
            fullLength: content.length
        };
    },

    // Create document preview
    async createDocumentPreview(file) {
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        
        switch (extension) {
            case '.txt':
            case '.md':
                return await this.createTextPreview(file);
            default:
                return this.createGenericPreview(file);
        }
    },

    // Create generic preview
    createGenericPreview(file) {
        return {
            type: 'generic',
            content: {
                name: file.name,
                size: Common.formatFileSize(file.size),
                type: file.type,
                lastModified: new Date(file.lastModified).toLocaleString()
            }
        };
    },

    // Get file type category
    getFileType(file) {
        const extension = '.' + file.name.split('.').pop().toLowerCase();
        
        for (const [type, config] of Object.entries(this.fileTypes)) {
            if (config.extensions.includes(extension) || config.mimeTypes.includes(file.type)) {
                return type;
            }
        }
        
        return 'unknown';
    },

    // Get image metadata
    async getImageMetadata(file) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                resolve({
                    width: img.width,
                    height: img.height,
                    aspectRatio: img.width / img.height
                });
            };
            img.onerror = () => resolve({});
            img.src = URL.createObjectURL(file);
        });
    },

    // Compress image file
    async compressImage(file, options = {}) {
        const {
            maxWidth = 1920,
            maxHeight = 1080,
            quality = 0.8,
            format = 'image/jpeg'
        } = options;

        return new Promise((resolve) => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                let { width, height } = img;
                
                // Calculate new dimensions
                if (width > maxWidth || height > maxHeight) {
                    const ratio = Math.min(maxWidth / width, maxHeight / height);
                    width *= ratio;
                    height *= ratio;
                }

                canvas.width = width;
                canvas.height = height;
                ctx.drawImage(img, 0, 0, width, height);
                
                canvas.toBlob(resolve, format, quality);
            };

            img.src = URL.createObjectURL(file);
        });
    },

    // Create file input programmatically
    createFileInput(options = {}) {
        const {
            accept = '*/*',
            multiple = false,
            onChange = () => {}
        } = options;

        const input = document.createElement('input');
        input.type = 'file';
        input.accept = accept;
        input.multiple = multiple;
        input.style.display = 'none';

        input.addEventListener('change', (e) => {
            const files = Array.from(e.target.files);
            onChange(files);
            input.remove(); // Clean up after use
        });

        document.body.appendChild(input);
        return input;
    },

    // Open file browser
    openFileBrowser(options = {}) {
        return new Promise((resolve) => {
            const input = this.createFileInput({
                ...options,
                onChange: resolve
            });
            input.click();
        });
    },

    // Batch process files
    async batchProcessFiles(files, processor, options = {}) {
        const {
            concurrency = 3,
            progressCallback = null
        } = options;

        const results = [];
        const total = files.length;
        let completed = 0;

        // Process files in batches
        for (let i = 0; i < files.length; i += concurrency) {
            const batch = files.slice(i, i + concurrency);
            const batchPromises = batch.map(async (file) => {
                try {
                    const result = await processor(file);
                    completed++;
                    if (progressCallback) {
                        progressCallback(completed / total * 100, `Processed ${completed}/${total} files`);
                    }
                    return { file, result, success: true };
                } catch (error) {
                    completed++;
                    if (progressCallback) {
                        progressCallback(completed / total * 100, `Processed ${completed}/${total} files`);
                    }
                    return { file, error: error.message, success: false };
                }
            });

            const batchResults = await Promise.all(batchPromises);
            results.push(...batchResults);
        }

        return results;
    },

    // Clean up object URLs
    cleanupObjectUrls(urls) {
        urls.forEach(url => {
            if (url && url.startsWith('blob:')) {
                URL.revokeObjectURL(url);
            }
        });
    }
};

// Export for use in other modules
window.FileHandler = FileHandler;