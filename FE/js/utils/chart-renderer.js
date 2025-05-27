// Enhanced Toolkit v2.0 - Chart Renderer Utility
// Chart.js integration for data visualization

const ChartRenderer = {
    // Chart instances storage
    charts: {},

    // Default chart options
    defaultOptions: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'top',
            },
            tooltip: {
                backgroundColor: 'rgba(0, 0, 0, 0.8)',
                titleColor: '#fff',
                bodyColor: '#fff',
                borderColor: '#3b82f6',
                borderWidth: 1
            }
        },
        scales: {
            y: {
                beginAtZero: true,
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            },
            x: {
                grid: {
                    color: 'rgba(0, 0, 0, 0.1)'
                }
            }
        }
    },

    // Color palettes
    colorPalettes: {
        primary: ['#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6', '#06b6d4', '#84cc16', '#f97316'],
        pastel: ['#93c5fd', '#fca5a5', '#6ee7b7', '#fcd34d', '#c4b5fd', '#67e8f9', '#bef264', '#fdba74'],
        dark: ['#1e40af', '#dc2626', '#059669', '#d97706', '#7c3aed', '#0891b2', '#65a30d', '#ea580c']
    },

    // Create word frequency chart
    createWordFrequencyChart(canvasId, commonWords, options = {}) {
        this.destroyChart(canvasId);

        const ctx = document.getElementById(canvasId).getContext('2d');
        const data = commonWords.slice(0, 15); // Top 15 words

        const chartData = {
            labels: data.map(item => item[0]),
            datasets: [{
                label: 'Frequency',
                data: data.map(item => item[1]),
                backgroundColor: this.colorPalettes.primary[0],
                borderColor: this.colorPalettes.dark[0],
                borderWidth: 1,
                borderRadius: 4,
                borderSkipped: false,
            }]
        };

        const config = {
            type: 'bar',
            data: chartData,
            options: {
                ...this.defaultOptions,
                ...options,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Most Common Words'
                    }
                },
                scales: {
                    ...this.defaultOptions.scales,
                    x: {
                        ...this.defaultOptions.scales.x,
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        };

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    },

    // Create document analysis overview chart
    createAnalysisOverviewChart(canvasId, analysisData, options = {}) {
        this.destroyChart(canvasId);

        const ctx = document.getElementById(canvasId).getContext('2d');
        
        const chartData = {
            labels: ['Words', 'Sentences', 'Paragraphs'],
            datasets: [{
                label: 'Count',
                data: [
                    analysisData.word_count || 0,
                    analysisData.sentence_count || 0,
                    analysisData.paragraph_count || 0
                ],
                backgroundColor: [
                    this.colorPalettes.primary[0],
                    this.colorPalettes.primary[1],
                    this.colorPalettes.primary[2]
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        };

        const config = {
            type: 'doughnut',
            data: chartData,
            options: {
                ...this.defaultOptions,
                ...options,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Document Structure'
                    }
                }
            }
        };

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    },

    // Create AI model comparison chart
    createModelComparisonChart(canvasId, aiResults, options = {}) {
        this.destroyChart(canvasId);

        const ctx = document.getElementById(canvasId).getContext('2d');
        const validResults = aiResults.filter(r => r && r.content && !r.content.includes('Error'));

        const chartData = {
            labels: validResults.map(r => r.model),
            datasets: [{
                label: 'Summary Length (characters)',
                data: validResults.map(r => r.content.length),
                backgroundColor: this.colorPalettes.pastel.slice(0, validResults.length),
                borderColor: this.colorPalettes.primary.slice(0, validResults.length),
                borderWidth: 2
            }]
        };

        const config = {
            type: 'bar',
            data: chartData,
            options: {
                ...this.defaultOptions,
                ...options,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'AI Model Summary Comparison'
                    }
                }
            }
        };

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    },

    // Create search results distribution chart
    createSearchResultsChart(canvasId, searchResults, options = {}) {
        this.destroyChart(canvasId);

        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // Count results by source
        const sourceCounts = {};
        searchResults.forEach(result => {
            const source = result.source || 'Unknown';
            sourceCounts[source] = (sourceCounts[source] || 0) + 1;
        });

        const chartData = {
            labels: Object.keys(sourceCounts),
            datasets: [{
                label: 'Results',
                data: Object.values(sourceCounts),
                backgroundColor: this.colorPalettes.primary.slice(0, Object.keys(sourceCounts).length),
                borderWidth: 2,
                borderColor: '#fff'
            }]
        };

        const config = {
            type: 'pie',
            data: chartData,
            options: {
                ...this.defaultOptions,
                ...options,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Search Results by Source'
                    }
                }
            }
        };

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    },

    // Create processing time chart
    createProcessingTimeChart(canvasId, processingData, options = {}) {
        this.destroyChart(canvasId);

        const ctx = document.getElementById(canvasId).getContext('2d');

        const chartData = {
            labels: processingData.map(d => d.filename),
            datasets: [{
                label: 'Processing Time (seconds)',
                data: processingData.map(d => d.processingTime || 0),
                backgroundColor: this.colorPalettes.primary[0],
                borderColor: this.colorPalettes.dark[0],
                borderWidth: 1,
                fill: false,
                tension: 0.4
            }]
        };

        const config = {
            type: 'line',
            data: chartData,
            options: {
                ...this.defaultOptions,
                ...options,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Document Processing Times'
                    }
                },
                scales: {
                    ...this.defaultOptions.scales,
                    x: {
                        ...this.defaultOptions.scales.x,
                        ticks: {
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        };

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    },

    // Create social media platform distribution chart
    createPlatformDistributionChart(canvasId, searchResults, options = {}) {
        this.destroyChart(canvasId);

        const ctx = document.getElementById(canvasId).getContext('2d');
        
        // Count by platform
        const platformCounts = {};
        searchResults.filter(r => r.is_social_media).forEach(result => {
            const platform = ImageSearch?.detectSocialPlatform(result.link) || 'Unknown';
            platformCounts[platform] = (platformCounts[platform] || 0) + 1;
        });

        const chartData = {
            labels: Object.keys(platformCounts),
            datasets: [{
                label: 'Posts Found',
                data: Object.values(platformCounts),
                backgroundColor: this.colorPalettes.primary.slice(0, Object.keys(platformCounts).length),
                borderWidth: 2,
                borderColor: '#fff'
            }]
        };

        const config = {
            type: 'doughnut',
            data: chartData,
            options: {
                ...this.defaultOptions,
                ...options,
                plugins: {
                    ...this.defaultOptions.plugins,
                    title: {
                        display: true,
                        text: 'Social Media Platform Distribution'
                    }
                }
            }
        };

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    },

    // Create animated progress chart
    createProgressChart(canvasId, progress = 0, options = {}) {
        this.destroyChart(canvasId);

        const ctx = document.getElementById(canvasId).getContext('2d');

        const chartData = {
            datasets: [{
                data: [progress, 100 - progress],
                backgroundColor: [
                    this.colorPalettes.primary[0],
                    'rgba(229, 231, 235, 0.3)'
                ],
                borderWidth: 0,
                cutout: '70%'
            }]
        };

        const config = {
            type: 'doughnut',
            data: chartData,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                animation: {
                    animateRotate: true,
                    duration: 1000
                },
                ...options
            }
        };

        this.charts[canvasId] = new Chart(ctx, config);
        return this.charts[canvasId];
    },

    // Update progress chart
    updateProgressChart(canvasId, newProgress) {
        const chart = this.charts[canvasId];
        if (chart) {
            chart.data.datasets[0].data = [newProgress, 100 - newProgress];
            chart.update('active');
        }
    },

    // Destroy specific chart
    destroyChart(canvasId) {
        if (this.charts[canvasId]) {
            this.charts[canvasId].destroy();
            delete this.charts[canvasId];
        }
    },

    // Destroy all charts
    destroyAllCharts() {
        Object.keys(this.charts).forEach(canvasId => {
            this.destroyChart(canvasId);
        });
    },

    // Get chart as image
    getChartImage(canvasId, format = 'png') {
        const chart = this.charts[canvasId];
        if (chart) {
            return chart.toBase64Image(format);
        }
        return null;
    },

    // Download chart as image
    downloadChart(canvasId, filename, format = 'png') {
        const imageUrl = this.getChartImage(canvasId, format);
        if (imageUrl) {
            const link = document.createElement('a');
            link.download = `${filename}.${format}`;
            link.href = imageUrl;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    },

    // Resize all charts (useful for responsive design)
    resizeAllCharts() {
        Object.values(this.charts).forEach(chart => {
            chart.resize();
        });
    },

    // Apply dark theme to chart
    applyDarkTheme(options = {}) {
        return {
            ...options,
            plugins: {
                ...options.plugins,
                legend: {
                    ...options.plugins?.legend,
                    labels: {
                        color: '#fff'
                    }
                }
            },
            scales: {
                ...options.scales,
                x: {
                    ...options.scales?.x,
                    ticks: {
                        ...options.scales?.x?.ticks,
                        color: '#fff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                y: {
                    ...options.scales?.y,
                    ticks: {
                        ...options.scales?.y?.ticks,
                        color: '#fff'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            }
        };
    }
};

// Handle window resize
window.addEventListener('resize', Common.debounce(() => {
    ChartRenderer.resizeAllCharts();
}, 250));

// Export for use in other modules
window.ChartRenderer = ChartRenderer;