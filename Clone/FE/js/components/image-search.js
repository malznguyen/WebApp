// Enhanced Toolkit v2.0 - Image Search Component
// Specialized image search functionality

const ImageSearch = {
    // Validate image file
    validateImageFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'];
        const maxSize = 10 * 1024 * 1024; // 10MB

        if (!validTypes.includes(file.type)) {
            return { valid: false, error: 'Invalid file type. Please select an image file.' };
        }

        if (file.size > maxSize) {
            return { valid: false, error: 'File too large. Maximum size is 10MB.' };
        }

        return { valid: true };
    },

    // Create image preview
    async createImagePreview(file) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const img = new Image();
                img.onload = () => {
                    resolve({
                        dataUrl: e.target.result,
                        width: img.width,
                        height: img.height,
                        aspectRatio: img.width / img.height
                    });
                };
                img.onerror = () => reject(new Error('Failed to load image'));
                img.src = e.target.result;
            };
            reader.onerror = () => reject(new Error('Failed to read file'));
            reader.readAsDataURL(file);
        });
    },

    // Compress image if needed
    async compressImage(file, maxWidth = 1920, maxHeight = 1080, quality = 0.8) {
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

                // Draw and compress
                ctx.drawImage(img, 0, 0, width, height);
                canvas.toBlob(resolve, file.type, quality);
            };

            img.src = URL.createObjectURL(file);
        });
    },

    // Extract EXIF data
    async extractImageMetadata(file) {
        try {
            const arrayBuffer = await file.arrayBuffer();
            const dataView = new DataView(arrayBuffer);
            
            // Basic EXIF parsing (simplified)
            const metadata = {
                name: file.name,
                size: file.size,
                type: file.type,
                lastModified: new Date(file.lastModified),
                dimensions: null
            };

            // Get image dimensions
            const preview = await this.createImagePreview(file);
            metadata.dimensions = {
                width: preview.width,
                height: preview.height,
                aspectRatio: preview.aspectRatio
            };

            return metadata;
        } catch (error) {
            console.warn('Failed to extract metadata:', error);
            return {
                name: file.name,
                size: file.size,
                type: file.type,
                lastModified: new Date(file.lastModified)
            };
        }
    },

    // Filter search results
    filterResults(results, filters) {
        let filtered = [...results];

        if (filters.socialOnly) {
            filtered = filtered.filter(result => result.is_social_media);
        }

        if (filters.platform && filters.platform !== 'all') {
            filtered = filtered.filter(result => {
                const platform = this.detectSocialPlatform(result.link);
                return platform === filters.platform;
            });
        }

        if (filters.minDate) {
            filtered = filtered.filter(result => {
                const resultDate = new Date(result.date || 0);
                return resultDate >= filters.minDate;
            });
        }

        return filtered;
    },

    // Detect social media platform
    detectSocialPlatform(url) {
        const platforms = {
            'facebook.com': 'facebook',
            'fb.com': 'facebook',
            'instagram.com': 'instagram',
            'twitter.com': 'twitter',
            'x.com': 'twitter',
            'linkedin.com': 'linkedin',
            'tiktok.com': 'tiktok',
            'youtube.com': 'youtube',
            'pinterest.com': 'pinterest',
            'snapchat.com': 'snapchat',
            'reddit.com': 'reddit'
        };

        const urlLower = url.toLowerCase();
        for (const [domain, platform] of Object.entries(platforms)) {
            if (urlLower.includes(domain)) {
                return platform;
            }
        }
        return 'unknown';
    },

    // Sort search results
    sortResults(results, sortBy = 'relevance') {
        const sorted = [...results];

        switch (sortBy) {
            case 'date':
                return sorted.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0));
            case 'title':
                return sorted.sort((a, b) => a.title.localeCompare(b.title));
            case 'platform':
                return sorted.sort((a, b) => {
                    const platformA = this.detectSocialPlatform(a.link);
                    const platformB = this.detectSocialPlatform(b.link);
                    return platformA.localeCompare(platformB);
                });
            case 'relevance':
            default:
                return sorted; // Keep original order for relevance
        }
    },

    // Generate search result statistics
    generateStats(results) {
        const stats = {
            total: results.length,
            socialMedia: results.filter(r => r.is_social_media).length,
            platforms: {},
            sources: {}
        };

        results.forEach(result => {
            // Count platforms
            if (result.is_social_media) {
                const platform = this.detectSocialPlatform(result.link);
                stats.platforms[platform] = (stats.platforms[platform] || 0) + 1;
            }

            // Count sources
            const source = result.source || 'Unknown';
            stats.sources[source] = (stats.sources[source] || 0) + 1;
        });

        return stats;
    },

    // Export search results
    exportResults(results, format = 'json') {
        const timestamp = new Date().toISOString().split('T')[0];
        const filename = `image_search_results_${timestamp}`;

        switch (format) {
            case 'json':
                const jsonData = JSON.stringify(results, null, 2);
                Common.downloadAsFile(jsonData, `${filename}.json`, 'application/json');
                break;

            case 'csv':
                const csvData = this.convertToCSV(results);
                Common.downloadAsFile(csvData, `${filename}.csv`, 'text/csv');
                break;

            case 'txt':
                const txtData = results.map(r => 
                    `${r.title}\n${r.link}\n${r.snippet}\n---\n`
                ).join('\n');
                Common.downloadAsFile(txtData, `${filename}.txt`, 'text/plain');
                break;
        }
    },

    // Convert results to CSV format
    convertToCSV(results) {
        const headers = ['Title', 'Link', 'Displayed Link', 'Snippet', 'Is Social Media', 'Platform', 'Source'];
        const rows = results.map(result => [
            result.title || '',
            result.link || '',
            result.displayed_link || '',
            result.snippet || '',
            result.is_social_media ? 'Yes' : 'No',
            result.is_social_media ? this.detectSocialPlatform(result.link) : '',
            result.source || ''
        ]);

        const csvContent = [headers, ...rows]
            .map(row => row.map(field => `"${field.replace(/"/g, '""')}"`).join(','))
            .join('\n');

        return csvContent;
    },

    // Batch image processing
    async processBatchImages(files, progressCallback) {
        const results = [];
        const total = files.length;

        for (let i = 0; i < files.length; i++) {
            const file = files[i];
            
            try {
                const validation = this.validateImageFile(file);
                if (!validation.valid) {
                    results.push({ file: file.name, error: validation.error });
                    continue;
                }

                const metadata = await this.extractImageMetadata(file);
                const preview = await this.createImagePreview(file);
                
                results.push({
                    file: file.name,
                    success: true,
                    metadata,
                    preview: preview.dataUrl
                });

                if (progressCallback) {
                    progressCallback((i + 1) / total * 100, `Processed ${i + 1}/${total} images`);
                }

            } catch (error) {
                results.push({ file: file.name, error: error.message });
            }
        }

        return results;
    }
};

// Export for use in other modules
window.ImageSearch = ImageSearch;