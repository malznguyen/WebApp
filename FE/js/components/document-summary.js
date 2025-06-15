// Enhanced Toolkit v2.0 - Document Summary Component
// Specialized document processing functionality

const DocumentSummary = {
    // Validate document file
    validateDocumentFile(file) {
        const validExtensions = ['.pdf', '.docx', '.txt', '.md'];
        const maxSize = 50 * 1024 * 1024; // 50MB

        const isValidType = validExtensions.some(ext => 
            file.name.toLowerCase().endsWith(ext)
        );

        if (!isValidType) {
            return { 
                valid: false, 
                error: 'Invalid file type. Supported: PDF, DOCX, TXT, MD' 
            };
        }

        if (file.size > maxSize) {
            return { 
                valid: false, 
                error: 'File too large. Maximum size is 50MB.' 
            };
        }

        return { valid: true };
    },

    // Get document type info
    getDocumentType(filename) {
        const ext = filename.toLowerCase().split('.').pop();
        const types = {
            'pdf': {
                name: 'PDF Document',
                icon: 'fas fa-file-pdf',
                color: 'text-red-600',
                description: 'Portable Document Format'
            },
            'docx': {
                name: 'Word Document',
                icon: 'fas fa-file-word',
                color: 'text-blue-600',
                description: 'Microsoft Word Document'
            },
            'txt': {
                name: 'Text File',
                icon: 'fas fa-file-alt',
                color: 'text-gray-600',
                description: 'Plain Text File'
            },
            'md': {
                name: 'Markdown File',
                icon: 'fas fa-markdown',
                color: 'text-purple-600',
                description: 'Markdown Document'
            }
        };
        return types[ext] || {
            name: 'Unknown',
            icon: 'fas fa-file',
            color: 'text-gray-400',
            description: 'Unknown file type'
        };
    },

    // Calculate processing time estimate
    estimateProcessingTime(fileSize, useAI) {
        const baseTime = 5; // seconds per MB
        const modelMultiplier = useAI ? 1.5 : 1;
        const sizeInMB = fileSize / (1024 * 1024);
        
        return Math.ceil(sizeInMB * baseTime * modelMultiplier);
    },

    // Generate processing queue summary
    generateQueueSummary(queue) {
        const summary = {
            totalFiles: queue.length,
            totalSize: queue.reduce((sum, item) => sum + item.size, 0),
            fileTypes: {},
            estimatedTime: 0
        };

        queue.forEach(item => {
            const type = this.getDocumentType(item.name);
            summary.fileTypes[type.name] = (summary.fileTypes[type.name] || 0) + 1;
        });

        return summary;
    },

    // Format analysis results for display
    formatAnalysisResults(analysis) {
        if (!analysis || analysis.error) {
            return { error: analysis?.error || 'Analysis failed' };
        }

        return {
            wordCount: this.formatNumber(analysis.word_count),
            sentenceCount: this.formatNumber(analysis.sentence_count),
            paragraphCount: this.formatNumber(analysis.paragraph_count),
            avgWordLength: this.formatDecimal(analysis.avg_word_length),
            avgSentenceLength: this.formatDecimal(analysis.avg_sentence_length),
            readingTime: this.calculateReadingTime(analysis.word_count),
            commonWords: analysis.common_words || []
        };
    },

    // Calculate estimated reading time
    calculateReadingTime(wordCount) {
        const averageWPM = 200; // words per minute
        const minutes = Math.ceil(wordCount / averageWPM);
        
        if (minutes < 1) return '< 1 min';
        if (minutes < 60) return `${minutes} min`;
        
        const hours = Math.floor(minutes / 60);
        const remainingMinutes = minutes % 60;
        return `${hours}h ${remainingMinutes}m`;
    },

    // Format numbers for display
    formatNumber(num) {
        if (typeof num !== 'number') return 'N/A';
        return num.toLocaleString();
    },

    // Format decimals for display
    formatDecimal(num, places = 1) {
        if (typeof num !== 'number') return 'N/A';
        return num.toFixed(places);
    },

    // Generate word cloud data
    generateWordCloudData(commonWords) {
        if (!Array.isArray(commonWords)) return [];
        
        return commonWords.map(([word, count], index) => ({
            text: word,
            size: Math.max(12, 32 - (index * 2)), // Decreasing size
            count: count,
            color: this.getWordColor(index)
        }));
    },

    // Get color for word cloud
    getWordColor(index) {
        const colors = [
            '#3b82f6', '#ef4444', '#10b981', '#f59e0b',
            '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'
        ];
        return colors[index % colors.length];
    },

    // Compare AI model results
    compareAIResults(results) {
        const comparison = {
            models: [],
            similarities: {},
            averageLength: 0,
            totalProcessingTime: 0
        };

        const validResults = results.filter(r => r && r.content && !r.content.includes('Error'));
        
        validResults.forEach(result => {
            comparison.models.push({
                name: result.model,
                length: result.content.length,
                wordCount: result.content.split(/\s+/).length,
                sentences: result.content.split(/[.!?]+/).length - 1
            });
        });

        // Calculate average length
        if (comparison.models.length > 0) {
            comparison.averageLength = Math.round(
                comparison.models.reduce((sum, m) => sum + m.length, 0) / comparison.models.length
            );
        }

        return comparison;
    },

    // Export document results
    exportDocumentResults(results, format = 'json') {
        const timestamp = new Date().toISOString().split('T')[0];
        const filename = `document_analysis_${timestamp}`;

        switch (format) {
            case 'json':
                const jsonData = JSON.stringify(results, null, 2);
                Common.downloadAsFile(jsonData, `${filename}.json`, 'application/json');
                break;

            case 'md':
                const mdData = this.convertToMarkdown(results);
                Common.downloadAsFile(mdData, `${filename}.md`, 'text/markdown');
                break;

            case 'txt':
                const txtData = this.convertToText(results);
                Common.downloadAsFile(txtData, `${filename}.txt`, 'text/plain');
                break;
        }
    },

    // Convert results to Markdown
    convertToMarkdown(results) {
        let markdown = '# Document Analysis Results\n\n';
        markdown += `**Generated:** ${new Date().toLocaleString()}\n\n`;

        if (results.analysis) {
            markdown += '## Analysis Summary\n\n';
            const analysis = this.formatAnalysisResults(results.analysis);
            markdown += `- **Words:** ${analysis.wordCount}\n`;
            markdown += `- **Sentences:** ${analysis.sentenceCount}\n`;
            markdown += `- **Paragraphs:** ${analysis.paragraphCount}\n`;
            markdown += `- **Reading Time:** ${analysis.readingTime}\n\n`;
        }

        if (results.ai_results && results.ai_results.length > 0) {
            markdown += '## AI Generated Summaries\n\n';
            results.ai_results.forEach(result => {
                if (result && result.content) {
                    markdown += `### ${result.model}\n\n`;
                    markdown += `${result.content}\n\n`;
                }
            });
        }

        return markdown;
    },

    // Convert results to plain text
    convertToText(results) {
        let text = 'DOCUMENT ANALYSIS RESULTS\n';
        text += '='.repeat(50) + '\n\n';
        text += `Generated: ${new Date().toLocaleString()}\n\n`;

        if (results.analysis) {
            text += 'ANALYSIS SUMMARY\n';
            text += '-'.repeat(20) + '\n';
            const analysis = this.formatAnalysisResults(results.analysis);
            text += `Words: ${analysis.wordCount}\n`;
            text += `Sentences: ${analysis.sentenceCount}\n`;
            text += `Paragraphs: ${analysis.paragraphCount}\n`;
            text += `Reading Time: ${analysis.readingTime}\n\n`;
        }

        if (results.ai_results && results.ai_results.length > 0) {
            text += 'AI GENERATED SUMMARIES\n';
            text += '-'.repeat(25) + '\n\n';
            results.ai_results.forEach(result => {
                if (result && result.content) {
                    text += `${result.model.toUpperCase()}\n`;
                    text += '-'.repeat(result.model.length) + '\n';
                    text += `${result.content}\n\n`;
                }
            });
        }

        return text;
    },

    // Generate processing report
    generateProcessingReport(queue, results) {
        const report = {
            summary: {
                totalFiles: queue.length,
                successfullyProcessed: results.filter(r => r.success).length,
                failed: results.filter(r => !r.success).length,
                totalProcessingTime: results.reduce((sum, r) => sum + (r.processingTime || 0), 0)
            },
            details: results.map(result => ({
                filename: result.filename,
                status: result.success ? 'Success' : 'Failed',
                processingTime: result.processingTime || 0,
                error: result.error || null
            }))
        };

        return report;
    },

    // Validate processing settings
    validateProcessingSettings(settings) {
        const errors = [];

        if (!settings.detailLevel || settings.detailLevel < 10 || settings.detailLevel > 90) {
            errors.push('Detail level must be between 10 and 90');
        }



        return {
            valid: errors.length === 0,
            errors
        };
    }
};

// Export for use in other modules
window.DocumentSummary = DocumentSummary;