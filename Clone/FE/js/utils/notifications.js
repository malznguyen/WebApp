// Enhanced Toolkit v2.0 - Notifications Utility
// Advanced notification system with queuing, persistence, and animations

const Notifications = {
    // Notification queue and settings
    queue: [],
    activeNotifications: new Map(),
    maxVisible: 5,
    defaultDuration: 5000,

    // Notification types with icons and colors
    types: {
        success: {
            icon: 'fas fa-check-circle',
            color: 'text-green-500',
            bgColor: 'bg-green-50',
            borderColor: 'border-green-200'
        },
        error: {
            icon: 'fas fa-times-circle',
            color: 'text-red-500',
            bgColor: 'bg-red-50',
            borderColor: 'border-red-200'
        },
        warning: {
            icon: 'fas fa-exclamation-triangle',
            color: 'text-yellow-500',
            bgColor: 'bg-yellow-50',
            borderColor: 'border-yellow-200'
        },
        info: {
            icon: 'fas fa-info-circle',
            color: 'text-blue-500',
            bgColor: 'bg-blue-50',
            borderColor: 'border-blue-200'
        },
        loading: {
            icon: 'fas fa-spinner fa-spin',
            color: 'text-gray-500',
            bgColor: 'bg-gray-50',
            borderColor: 'border-gray-200'
        }
    },

    // Show notification
    show(type, title, message, options = {}) {
        const notification = this.createNotification(type, title, message, options);

        if (this.activeNotifications.size >= this.maxVisible) {
            this.queue.push(notification);
        } else {
            this.displayNotification(notification);
        }

        return notification.id;
    },

    // Convenience methods
    success(title, message, options = {}) {
        return this.show('success', title, message, options);
    },

    error(title, message, options = {}) {
        return this.show('error', title, message, options);
    },

    warning(title, message, options = {}) {
        return this.show('warning', title, message, options);
    },

    info(title, message, options = {}) {
        return this.show('info', title, message, options);
    },

    loading(title, message, options = {}) {
        return this.show('loading', title, message, {
            ...options,
            persistent: true,
            duration: 0
        });
    },

    // Create notification object
    createNotification(type, title, message, options = {}) {
        const {
            duration = this.defaultDuration,
            persistent = false,
            actions = [],
            onClick = null,
            onClose = null,
            html = false,
            position = 'top-right'
        } = options;

        return {
            id: Common.generateId(), // Assuming Common.generateId() is defined elsewhere
            type,
            title,
            message,
            duration,
            persistent,
            actions,
            onClick,
            onClose,
            html,
            position,
            createdAt: new Date(),
            element: null,
            timeout: null
        };
    },

    // Display notification in DOM
    displayNotification(notification) {
        const element = this.createNotificationElement(notification);
        notification.element = element;

        this.activeNotifications.set(notification.id, notification);
        this.appendToContainer(element, notification.position);

        // Animate in
        requestAnimationFrame(() => {
            element.classList.add('animate-slideIn');
        });

        // Set auto-dismiss timer
        if (!notification.persistent && notification.duration > 0) {
            notification.timeout = setTimeout(() => {
                this.dismiss(notification.id);
            }, notification.duration);
        }

        // Add click handler
        if (notification.onClick) {
            element.addEventListener('click', notification.onClick);
        }
    },

    // Create notification DOM element
    createNotificationElement(notification) {
        const typeConfig = this.types[notification.type];
        const div = document.createElement('div');

        div.className = `notification-item max-w-sm w-full ${typeConfig.bgColor} ${typeConfig.borderColor} border shadow-lg rounded-lg pointer-events-auto overflow-hidden`;
        div.setAttribute('data-notification-id', notification.id);

        const content = `
            <div class="p-4">
                <div class="flex items-start">
                    <div class="flex-shrink-0">
                        <i class="${typeConfig.icon} ${typeConfig.color} text-lg"></i>
                    </div>
                    <div class="ml-3 flex-1">
                        <div class="flex items-baseline whitespace-nowrap">
                            <span class="text-sm font-semibold text-gray-900">${notification.html ? notification.title : this.escapeHtml(notification.title)}</span>
                            <span class="ml-2 text-sm text-gray-600">${notification.html ? notification.message : this.escapeHtml(notification.message)}</span>
                        </div>
                        ${this.renderActions(notification.actions)}
                    </div>
                    <div class="ml-4 flex-shrink-0 flex">
                        <button class="notification-close inline-flex text-gray-400 hover:text-gray-600 focus:outline-none transition-colors">
                            <i class="fas fa-times text-sm"></i>
                        </button>
                    </div>
                </div>
            </div>
            ${this.renderProgressBar(notification)}
        `;

        div.innerHTML = content;

        // Add close button handler
        const closeBtn = div.querySelector('.notification-close');
        if (closeBtn) { // Good practice to check if element exists
            closeBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                this.dismiss(notification.id);
            });
        }

        // Add action button handlers
        notification.actions.forEach((action, index) => {
            const btn = div.querySelector(`[data-action-index="${index}"]`);
            if (btn && action.handler) {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    action.handler(notification);
                    if (action.dismissAfter !== false) {
                        this.dismiss(notification.id);
                    }
                });
            }
        });

        return div;
    },

    // Render action buttons
    renderActions(actions) {
        if (!actions || actions.length === 0) return '';

        return `
            <div class="mt-3 flex space-x-2">
                ${actions.map((action, index) => `
                    <button data-action-index="${index}"
                            class="text-sm font-medium ${action.style || 'text-blue-600 hover:text-blue-500'} transition-colors">
                        ${action.label}
                    </button>
                `).join('')}
            </div>
        `;
    },

    // Render progress bar for loading notifications
    renderProgressBar(notification) {
        if (notification.type !== 'loading' || !notification.progress) return '';

        return `
            <div class="px-4 pb-2">
                <div class="w-full bg-gray-200 rounded-full h-2">
                    <div class="bg-blue-600 h-2 rounded-full transition-all duration-300"
                         style="width: ${notification.progress}%"></div>
                </div>
            </div>
        `;
    },

    // Get or create notification container
    getContainer(position = 'top-right') {
        const containerId = `notifications-${position}`;
        let container = document.getElementById(containerId);

        if (!container) {
            container = document.createElement('div');
            container.id = containerId;
            container.className = `fixed z-50 space-y-3 pointer-events-none ${this.getPositionClasses(position)}`;
            document.body.appendChild(container);
        }

        return container;
    },

    // Get CSS classes for position
    getPositionClasses(position) {
        const positions = {
            'top-right': 'top-4 right-4',
            'top-left': 'top-4 left-4',
            'top-center': 'top-4 left-1/2 transform -translate-x-1/2',
            'bottom-right': 'bottom-4 right-4',
            'bottom-left': 'bottom-4 left-4',
            'bottom-center': 'bottom-4 left-1/2 transform -translate-x-1/2'
        };
        return positions[position] || positions['top-right'];
    },

    // Append notification to container
    appendToContainer(element, position) {
        const container = this.getContainer(position);
        container.appendChild(element);
    },

    // Dismiss notification
    dismiss(notificationId) {
        const notification = this.activeNotifications.get(notificationId);
        if (!notification) return;

        // Clear timeout
        if (notification.timeout) {
            clearTimeout(notification.timeout);
        }

        // Animate out
        if (notification.element) {
            notification.element.style.transition = 'all 0.3s ease-out';
            // Example animation, adjust as needed. The original used 'animate-slideOut' which might need CSS definition
            notification.element.style.transform = 'translateX(100%)'; // Or use a class that triggers animation
            notification.element.style.opacity = '0';

            setTimeout(() => {
                if (notification.element && notification.element.parentNode) {
                    notification.element.parentNode.removeChild(notification.element);
                }
            }, 300); // Duration of animation
        }

        // Call onClose callback
        if (notification.onClose) {
            notification.onClose(notification);
        }

        // Remove from active notifications
        this.activeNotifications.delete(notificationId);

        // Show next queued notification
        this.processQueue();
    },

    // Process notification queue
    processQueue() {
        if (this.queue.length > 0 && this.activeNotifications.size < this.maxVisible) {
            const nextNotification = this.queue.shift();
            this.displayNotification(nextNotification);
        }
    },

    // Update existing notification
    update(notificationId, updates) {
        const notification = this.activeNotifications.get(notificationId);
        if (!notification) return false;

        // Update notification properties
        Object.assign(notification, updates);

        // Update DOM element
        if (notification.element) {
            const titleEl = notification.element.querySelector('.font-semibold'); // Assuming this class exists for title
            const messageEl = notification.element.querySelector('.text-gray-600'); // Assuming this class exists for message

            if (titleEl && updates.title !== undefined) {
                titleEl.textContent = updates.title;
            }

            if (messageEl && updates.message !== undefined) {
                messageEl.textContent = updates.message;
            }

            // Update progress bar if it's a loading notification
            if (updates.progress !== undefined) {
                const progressBar = notification.element.querySelector('.bg-blue-600'); // Class from renderProgressBar
                if (progressBar) {
                    progressBar.style.width = `${updates.progress}%`;
                }
            }
        }

        return true;
    },

    // Update progress for loading notification
    updateProgress(notificationId, progress, message) {
        const updates = { progress };
        if (message !== undefined) {
            updates.message = message;
        }
        return this.update(notificationId, updates);
    },

    // Dismiss all notifications
    dismissAll() {
        const ids = Array.from(this.activeNotifications.keys());
        ids.forEach(id => this.dismiss(id));
        this.queue = [];
    },

    // Get notification by ID
    get(notificationId) {
        return this.activeNotifications.get(notificationId);
    },

    // Check if notification exists
    exists(notificationId) {
        return this.activeNotifications.has(notificationId);
    },

    // Get all active notifications
    getAll() {
        return Array.from(this.activeNotifications.values());
    },

    // Clear expired notifications
    clearExpired() {
        const now = new Date();
        const expired = [];

        this.activeNotifications.forEach((notification, id) => {
            if (!notification.persistent && notification.duration > 0) {
                const expireTime = new Date(notification.createdAt.getTime() + notification.duration);
                if (now > expireTime) {
                    expired.push(id);
                }
            }
        });

        expired.forEach(id => this.dismiss(id));
    },

    // Set global configuration
    configure(options = {}) {
        if (options.maxVisible !== undefined) {
            this.maxVisible = options.maxVisible;
        }
        if (options.defaultDuration !== undefined) {
            this.defaultDuration = options.defaultDuration;
        }
    },

    // Utility methods
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    },

    // Persist notifications to localStorage
    persistNotifications() {
        const notifications = this.getAll().map(n => ({
            id: n.id,
            type: n.type,
            title: n.title,
            message: n.message,
            createdAt: n.createdAt,
            persistent: n.persistent
        }));

        // Assuming Common.storage is defined elsewhere
        if (window.Common && Common.storage) {
            Common.storage.set('notifications', notifications);
        } else {
            console.warn('Common.storage not available for persisting notifications.');
            // Fallback or handle error appropriately
        }
    },

    // Restore notifications from localStorage
    restoreNotifications() {
        // Assuming Common.storage is defined elsewhere
        if (window.Common && Common.storage) {
            const stored = Common.storage.get('notifications', []);
            const cutoff = new Date(Date.now() - 24 * 60 * 60 * 1000); // 24 hours ago

            stored.forEach(notification => {
                if (new Date(notification.createdAt) > cutoff) {
                    this.show(
                        notification.type,
                        notification.title,
                        notification.message,
                        { persistent: notification.persistent }
                    );
                }
            });
        } else {
            console.warn('Common.storage not available for restoring notifications.');
        }
    },

    // Show system notification (browser notification API)
    async showSystemNotification(title, message, options = {}) {
        if (!('Notification' in window)) {
            console.warn('Browser does not support notifications');
            return false;
        }

        let permission = Notification.permission;

        if (permission === 'default') {
            permission = await Notification.requestPermission();
        }

        if (permission === 'granted') {
            const notification = new Notification(title, {
                body: message,
                icon: options.icon || '/favicon.ico', // Ensure favicon.ico exists or provide a valid path
                badge: options.badge,
                tag: options.tag,
                requireInteraction: options.requireInteraction || false,
                silent: options.silent || false
            });

            if (options.onClick) {
                notification.onclick = options.onClick;
            }

            return notification;
        }

        return false;
    },

    // Batch notifications
    showBatch(notifications) {
        notifications.forEach(({ type, title, message, options }) => {
            this.show(type, title, message, options);
        });
    }
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    // Restore persisted notifications
    Notifications.restoreNotifications();

    // Clean up expired notifications periodically
    setInterval(() => {
        Notifications.clearExpired();
    }, 60000); // Every minute

    // Define Common.generateId and Common.storage if they are not globally available
    // This is a placeholder. You should have these defined in your actual 'Common' utility object.
    if (typeof Common === 'undefined') {
        window.Common = {
            generateId: () => `id_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            storage: {
                get: (key, defaultValue) => {
                    try {
                        const item = localStorage.getItem(key);
                        return item ? JSON.parse(item) : defaultValue;
                    } catch (e) {
                        console.error("Error reading from localStorage", e);
                        return defaultValue;
                    }
                },
                set: (key, value) => {
                    try {
                        localStorage.setItem(key, JSON.stringify(value));
                    } catch (e) {
                        console.error("Error writing to localStorage", e);
                    }
                }
            }
        };
    }
});

// Persist notifications before page unload
window.addEventListener('beforeunload', () => {
    Notifications.persistNotifications();
});

// Export for use in other modules
window.Notifications = Notifications;