const ImageVision = {
    // Vision settings và state
    settings: {
        language: 'vietnamese',
        detailLevel: 'detailed',
        autoAnalyze: false
    },

    // Validate settings for vision analysis
    validateVisionSettings(settings) {
        const errors = [];
        
        if (!['vietnamese', 'english'].includes(settings.language)) {
            errors.push('Language must be vietnamese or english');
        }
        
        if (!['brief', 'detailed', 'extensive'].includes(settings.detailLevel)) {
            errors.push('Detail level must be brief, detailed, or extensive');
        }
        
        return {
            valid: errors.length === 0,
            errors
        };
    },

    // Format vision result for display
    formatVisionResult(result) {
        if (!result || !result.success) {
            return {
                error: result?.error || 'Vision analysis failed',
                description: null
            };
        }

        return {
            description: result.description,
            metadata: {
                language: result.language,
                detailLevel: result.detail_level,
                wordCount: result.text_metrics?.word_count || 0,
                charCount: result.text_metrics?.char_count || 0,
                processingTime: result.processing_time_seconds || 0,
                imageInfo: result.image_metadata,
                costEstimate: result.api_usage?.cost_estimate || 0,
                tokensUsed: result.api_usage?.total_tokens || 0
            }
        };
    },

    // Export vision result
    exportVisionResult(result, filename = 'vision_analysis') {
        const timestamp = new Date().toISOString().split('T')[0];
        const exportData = {
            analysis_date: new Date().toISOString(),
            filename: result.filename || 'unknown',
            description: result.description,
            metadata: result.metadata,
            settings: {
                language: result.metadata.language,
                detail_level: result.metadata.detailLevel
            }
        };

        const jsonData = JSON.stringify(exportData, null, 2);
        Common.downloadAsFile(jsonData, `${filename}_${timestamp}.json`, 'application/json');
    },

    // Get cost information
    getCostInfo(detailLevel) {
        const costs = {
            'brief': { min: 0.001, max: 0.003, description: 'Ngắn gọn (1-2 câu)' },
            'detailed': { min: 0.003, max: 0.008, description: 'Chi tiết (3-5 đoạn)' },
            'extensive': { min: 0.008, max: 0.015, description: 'Toàn diện (phân tích sâu)' }
        };
        
        return costs[detailLevel] || costs['detailed'];
    },

    // Create vision settings panel HTML
    createSettingsPanel() {
        return `
            <div class="vision-settings-panel p-4 bg-gray-50 rounded-lg border">
                <h4 class="font-semibold text-gray-800 mb-3 flex items-center">
                    <i class="fas fa-eye mr-2 text-blue-500"></i>
                    Vision Analysis Settings
                </h4>
                
                <div class="space-y-3">
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Language</label>
                        <select id="visionLanguage" class="form-input w-full text-sm">
                            <option value="vietnamese">Tiếng Việt</option>
                            <option value="english">English</option>
                        </select>
                    </div>
                    
                    <div>
                        <label class="block text-sm font-medium text-gray-700 mb-1">Detail Level</label>
                        <select id="visionDetailLevel" class="form-input w-full text-sm">
                            <option value="brief">Brief (Quick, $0.001-$0.003)</option>
                            <option value="detailed" selected>Detailed (Standard, $0.003-$0.008)</option>
                            <option value="extensive">Extensive (Comprehensive, $0.008-$0.015)</option>
                        </select>
                    </div>
                    
                    <div class="flex items-center space-x-2">
                        <input type="checkbox" id="autoAnalyzeVision" class="form-checkbox">
                        <label for="autoAnalyzeVision" class="text-sm text-gray-700">Auto-analyze when image selected</label>
                    </div>
                </div>
            </div>
        `;
    }
};

// Export for use in main app
window.ImageVision = ImageVision;