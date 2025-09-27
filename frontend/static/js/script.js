class TumorScopeApp {
    constructor() {
        this.API_URL = 'http://localhost:8001';
        this.currentFile = null;
        this.isAnalyzing = false;
        
        this.initializeApp();
        this.setupEventListeners();
        this.startLoadingSequence();
    }

    // Initialize app elements
    initializeApp() {
        this.elements = {
            // Loading
            loadingScreen: document.getElementById('loadingScreen'),
            appContainer: document.getElementById('appContainer'),
            
            // Upload elements
            dropZone: document.getElementById('dropZone'),
            fileInput: document.getElementById('fileInput'),
            preview: document.getElementById('preview'),
            uploadContent: document.getElementById('uploadContent'),
            
            // Buttons
            analyzeBtn: document.getElementById('analyzeBtn'),
            clearBtn: document.getElementById('clearBtn'),
            infoBtn: document.getElementById('infoBtn'),
            modalClose: document.getElementById('modalClose'),
            
            // Progress
            analysisProgress: document.getElementById('analysisProgress'),
            progressFill: document.getElementById('progressFill'),
            progressText: document.getElementById('progressText'),
            
            // Results
            resultSection: document.getElementById('resultSection'),
            
            // Modal
            infoModal: document.getElementById('infoModal')
        };
    }

    // Setup all event listeners
    setupEventListeners() {
        // File upload events
        this.setupFileUploadEvents();
        
        // Button events
        this.elements.analyzeBtn.addEventListener('click', () => this.analyzeImage());
        this.elements.clearBtn.addEventListener('click', () => this.clearUpload());
        
        // Modal events
        this.elements.infoBtn.addEventListener('click', () => this.showModal());
        this.elements.modalClose.addEventListener('click', () => this.hideModal());
        
        // Click outside modal to close
        this.elements.infoModal.addEventListener('click', (e) => {
            if (e.target === this.elements.infoModal) this.hideModal();
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.hideModal();
        });
    }

    // Setup file upload drag and drop functionality
    setupFileUploadEvents() {
        const dropZone = this.elements.dropZone;
        const fileInput = this.elements.fileInput;

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop zone when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.highlight(dropZone), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => this.unhighlight(dropZone), false);
        });

        // Handle dropped files
        dropZone.addEventListener('drop', (e) => this.handleDrop(e), false);

        // Handle file input change
        fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));

        // Handle click on drop zone
        dropZone.addEventListener('click', (e) => {
            if (e.target !== fileInput) {
                fileInput.click();
            }
        });
    }

    // Utility functions for drag and drop
    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    highlight(element) {
        element.classList.add('drag-over');
    }

    unhighlight(element) {
        element.classList.remove('drag-over');
    }

    handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        this.handleFiles(files);
    }

    // Handle file selection
    handleFiles(files) {
        if (files.length === 0) return;

        const file = files[0];
        
        // Validate file type
        if (!this.validateFile(file)) {
            this.showNotification('Please select a valid image file (JPG, PNG, DICOM)', 'error');
            return;
        }

        // Validate file size (10MB limit)
        if (file.size > 10 * 1024 * 1024) {
            this.showNotification('File size must be less than 10MB', 'error');
            return;
        }

        this.currentFile = file;
        this.displayPreview(file);
        this.showUploadActions();
    }

    // Validate file type
    validateFile(file) {
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'application/dicom'];
        const validExtensions = ['.jpg', '.jpeg', '.png', '.dcm'];
        
        return validTypes.includes(file.type) || 
               validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    }

    // Display image preview
    displayPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.elements.preview.src = e.target.result;
            this.elements.preview.classList.remove('hidden');
            this.elements.uploadContent.classList.add('hidden');
            
            // Add smooth transition
            setTimeout(() => {
                this.elements.preview.style.opacity = '1';
            }, 100);
        };
        reader.readAsDataURL(file);
    }

    // Show upload action buttons
    showUploadActions() {
        this.elements.analyzeBtn.classList.remove('hidden');
        this.elements.clearBtn.classList.remove('hidden');
        
        // Animate buttons
        setTimeout(() => {
            this.elements.analyzeBtn.style.transform = 'translateY(0)';
            this.elements.clearBtn.style.transform = 'translateY(0)';
        }, 100);
    }

    // Clear upload
    clearUpload() {
        this.currentFile = null;
        this.elements.preview.classList.add('hidden');
        this.elements.preview.src = '';
        this.elements.uploadContent.classList.remove('hidden');
        this.elements.analyzeBtn.classList.add('hidden');
        this.elements.clearBtn.classList.add('hidden');
        this.elements.resultSection.classList.add('hidden');
        this.elements.fileInput.value = '';
    }

    // Start loading sequence
    startLoadingSequence() {
        // Hide loading screen after 2.5 seconds
        setTimeout(() => {
            this.elements.loadingScreen.style.opacity = '0';
            this.elements.appContainer.style.opacity = '1';
            
            setTimeout(() => {
                this.elements.loadingScreen.style.display = 'none';
            }, 500);
        }, 2500);
    }

    // Show progress animation
    showProgress() {
        this.elements.analysisProgress.classList.remove('hidden');
        this.elements.resultSection.classList.add('hidden');
        
        // Animate progress
        this.animateProgress();
    }

    // Animate progress bar
    animateProgress() {
        const steps = [
            { width: 20, text: 'Loading neural networks...' },
            { width: 40, text: 'Preprocessing image...' },
            { width: 60, text: 'Analyzing brain tissue...' },
            { width: 80, text: 'Detecting anomalies...' },
            { width: 95, text: 'Generating report...' },
            { width: 100, text: 'Analysis complete!' }
        ];

        let currentStep = 0;
        const progressInterval = setInterval(() => {
            if (currentStep >= steps.length) {
                clearInterval(progressInterval);
                return;
            }

            const step = steps[currentStep];
            this.elements.progressFill.style.width = `${step.width}%`;
            this.elements.progressText.textContent = step.text;
            
            currentStep++;
        }, 800);
    }

    // Hide progress
    hideProgress() {
        this.elements.analysisProgress.classList.add('hidden');
    }

    // Analyze image
    async analyzeImage() {
        if (!this.currentFile || this.isAnalyzing) return;

        this.isAnalyzing = true;
        this.elements.analyzeBtn.disabled = true;
        this.elements.analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Analyzing...</span>';
        
        this.showProgress();

        const formData = new FormData();
        formData.append('file', this.currentFile);

        try {
            const response = await fetch(`${this.API_URL}/predict`, {
                method: 'POST',
                body: formData,
                headers: {
                    'Accept': 'application/json',
                },
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Server error: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            
            // Wait for progress animation to complete
            setTimeout(() => {
                this.hideProgress();
                this.displayResults(result);
            }, 2000);
            
        } catch (error) {
            console.error('Error:', error);
            this.hideProgress();
            this.showNotification('Analysis failed. Please try again.', 'error');
        } finally {
            this.isAnalyzing = false;
            this.elements.analyzeBtn.disabled = false;
            this.elements.analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> <span>Analyze MRI Scan</span>';
        }
    }

    // Display comprehensive results
    displayResults(data) {
        if (data.error) {
            this.showNotification(`Error: ${data.error}`, 'error');
            return;
        }

        const tumor = data.tumor_details;
        const recommendation = data.recommendation;
        
        // Create results HTML
        const resultsHTML = this.generateResultsHTML(data, tumor, recommendation);
        
        this.elements.resultSection.innerHTML = resultsHTML;
        this.elements.resultSection.classList.remove('hidden');
        
        // Animate results appearance
        this.animateResults();
        
        // Setup result interactions
        this.setupResultInteractions();
    }

    // Generate comprehensive results HTML
    generateResultsHTML(data, tumor, recommendation) {
        const confidencePercentage = (data.confidence * 100).toFixed(1);
        const severityClass = this.getSeverityClass(tumor.severity);
        
        return `
            <div class="result-header">
                <h2 class="result-title">Analysis Complete</h2>
                <p class="result-subtitle">Comprehensive AI-powered diagnostic report</p>
            </div>
            
            <div class="result-grid">
                <!-- Diagnosis Card -->
                <div class="result-card diagnosis-card">
                    <div class="card-header">
                        <div class="card-icon diagnosis-icon">
                            <i class="fas fa-stethoscope"></i>
                        </div>
                        <h3 class="card-title">Diagnosis</h3>
                    </div>
                    <div class="card-content">
                        <div class="diagnosis-result">${data.class}</div>
                        <div class="severity-badge ${severityClass}">${tumor.severity} Severity</div>
                        <p class="diagnosis-description">${tumor.description}</p>
                    </div>
                </div>
                
                <!-- Confidence Card -->
                <div class="result-card confidence-card">
                    <div class="card-header">
                        <div class="card-icon confidence-icon">
                            <i class="fas fa-chart-line"></i>
                        </div>
                        <h3 class="card-title">Confidence Analysis</h3>
                    </div>
                    <div class="card-content">
                        <div class="confidence-text">${confidencePercentage}%</div>
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: ${confidencePercentage}%"></div>
                        </div>
                        <p><strong>Confidence Level:</strong> ${recommendation.confidence_level}</p>
                        <p><strong>Reliability:</strong> ${this.getReliabilityText(data.confidence)}</p>
                    </div>
                </div>
            </div>
            
            <div class="result-grid">
                <!-- Medical Details Card -->
                <div class="result-card details-card">
                    <div class="card-header">
                        <div class="card-icon details-icon">
                            <i class="fas fa-file-medical"></i>
                        </div>
                        <h3 class="card-title">Medical Information</h3>
                    </div>
                    <div class="card-content">
                        <div class="medical-section">
                            <h4>Tumor Type</h4>
                            <p>${tumor.type}</p>
                        </div>
                        
                        ${tumor.symptoms.length > 0 ? `
                        <div class="medical-section">
                            <h4>Common Symptoms</h4>
                            <ul class="detail-list">
                                ${tumor.symptoms.map(symptom => `<li>${symptom}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        ${tumor.subtypes.length > 0 ? `
                        <div class="medical-section">
                            <h4>Subtypes</h4>
                            <ul class="detail-list">
                                ${tumor.subtypes.map(subtype => `<li>${subtype}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        <div class="medical-section">
                            <h4>Prognosis</h4>
                            <p>${tumor.prognosis}</p>
                        </div>
                    </div>
                </div>
                
                <!-- Recommendations Card -->
                <div class="result-card recommendations-card">
                    <div class="card-header">
                        <div class="card-icon recommendations-icon">
                            <i class="fas fa-clipboard-list"></i>
                        </div>
                        <h3 class="card-title">Recommendations</h3>
                    </div>
                    <div class="card-content">
                        <div class="urgency-section">
                            <h4>Priority Level</h4>
                            <p><strong>${tumor.urgency}</strong></p>
                        </div>
                        
                        ${tumor.next_steps.length > 0 ? `
                        <div class="next-steps">
                            <h4>Recommended Next Steps</h4>
                            <ul class="detail-list">
                                ${tumor.next_steps.map(step => `<li>${step}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        ${tumor.treatment_options.length > 0 ? `
                        <div class="treatment-options">
                            <h4>Treatment Options</h4>
                            <ul class="detail-list">
                                ${tumor.treatment_options.map(treatment => `<li>${treatment}</li>`).join('')}
                            </ul>
                        </div>
                        ` : ''}
                        
                        <div class="disclaimer">
                            <h4>⚠️ Important Disclaimer</h4>
                            <p>${recommendation.medical_disclaimer}</p>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Get severity class for styling
    getSeverityClass(severity) {
        switch (severity.toLowerCase()) {
            case 'high': return 'severity-high';
            case 'moderate': case 'low to moderate': return 'severity-moderate';
            case 'none': return 'severity-none';
            default: return 'severity-moderate';
        }
    }

    // Get reliability text based on confidence
    getReliabilityText(confidence) {
        if (confidence >= 0.95) return 'Very High';
        if (confidence >= 0.85) return 'High';
        if (confidence >= 0.70) return 'Moderate';
        if (confidence >= 0.50) return 'Low';
        return 'Very Low';
    }

    // Animate results appearance
    animateResults() {
        const cards = this.elements.resultSection.querySelectorAll('.result-card');
        
        cards.forEach((card, index) => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(30px)';
            
            setTimeout(() => {
                card.style.transition = 'all 0.6s ease';
                card.style.opacity = '1';
                card.style.transform = 'translateY(0)';
            }, index * 200);
        });
    }

    // Setup result interactions
    setupResultInteractions() {
        // Add smooth scrolling to results
        this.elements.resultSection.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }

    // Show modal
    showModal() {
        this.elements.infoModal.classList.remove('hidden');
        this.elements.infoModal.classList.add('active');
    }

    // Hide modal
    hideModal() {
        this.elements.infoModal.classList.remove('active');
        setTimeout(() => {
            this.elements.infoModal.classList.add('hidden');
        }, 300);
    }

    // Show notification
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <i class="fas ${type === 'error' ? 'fa-exclamation-triangle' : 'fa-info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => notification.classList.add('show'), 100);
        
        // Remove after 5 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new TumorScopeApp();
});

// Add notification styles dynamically
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(10, 15, 28, 0.95);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transform: translateX(100%);
        transition: transform 0.3s ease;
        z-index: 10001;
        max-width: 400px;
    }
    
    .notification.show {
        transform: translateX(0);
    }
    
    .notification-content {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .notification-error {
        border-left: 4px solid #fa709a;
    }
    
    .notification-info {
        border-left: 4px solid #667eea;
    }
`;
document.head.appendChild(notificationStyles);