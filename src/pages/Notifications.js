import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom'; // Import Link for navigation
import Navbar from "../components/Navbar";
import './Notifications.css';

const Notifications = () => {
    const [notifications, setNotifications] = useState([]);
    const [selectedNotification, setSelectedNotification] = useState(null);

    useEffect(() => {
        const storedNotifications = JSON.parse(localStorage.getItem('notifications')) || [];
        setNotifications(storedNotifications);
    }, []);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/notifications');

        ws.onopen = () => {
            console.log('WebSocket connection established');
        };

        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            const newNotification = {
                recordId: data.record_id,
                reportId: data.report_id,
                plateNumber: data.plate_number,
                detectionTime: data.timestamp, // Map timestamp to detectionTime
                detectionLocation: data.location,
                message: data.message,
                carImage: data.Car_Image, // Add carImage from WebSocket data
                seen: false,
            };
        
            setNotifications((prevNotifications) => {
                const updatedNotifications = [newNotification, ...prevNotifications].sort(
                    (a, b) => b.recordId - a.recordId
                );
                localStorage.setItem('notifications', JSON.stringify(updatedNotifications));
                return updatedNotifications;
            });
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        ws.onclose = () => {
            console.log('WebSocket connection closed');
        };

        return () => {
            ws.close();
        };
    }, []);

    const handleNotificationClick = (notification) => {
        setNotifications((prevNotifications) =>
            prevNotifications.map((n) =>
                n.recordId === notification.recordId ? { ...n, seen: true } : n
            )
        );
        setSelectedNotification(notification);
    };

    const handleRemoveNotification = (recordId) => {
        setNotifications((prevNotifications) => {
            const updatedNotifications = prevNotifications.filter((n) => n.recordId !== recordId);
            localStorage.setItem('notifications', JSON.stringify(updatedNotifications));
            return updatedNotifications;
        });

        if (selectedNotification && selectedNotification.recordId === recordId) {
            setSelectedNotification(null);
        }
    };

    const getNotificationIndex = (notification) => {
        if (!notification) return ''; // Handle case where no notification is selected
        const index = notifications.findIndex(n => n.recordId === notification.recordId);
        return notifications.length - index;
    };

    return (
        <div className="notifications-page">
            <Navbar />
            <div className="notifications-container">
                {notifications.length > 0 ? (
                    <div className="notifications-panels">
                        <div className="notifications-left-panel">
                            {selectedNotification ? (
                                <div className="notification-details">
                                    <h3>Notification {getNotificationIndex(selectedNotification)}</h3>
                                    <p className="detail-item">
                                        <span className="detail-label">Report ID:</span>
                                        <span className="detail-value">{selectedNotification.reportId}</span>
                                    </p>
                                    <p className="detail-item">
                                        <span className="detail-label">Plate Number:</span>
                                        <span className="detail-value">{selectedNotification.plateNumber}</span>
                                    </p>
                                    <p className="detail-item">
                                        <span className="detail-label">Detection Time:</span>
                                      <span className="detail-value timestamp">{selectedNotification.detectionTime}</span>
                                    </p>
                                    <p className="detail-item">
                                        <span className="detail-label">Location:</span>
                                        <span className="detail-value location">
                                        <a
                                          href={selectedNotification.detectionLocation}
                                                target="_blank"
                                                rel="noopener noreferrer"
                                                >
                                            {selectedNotification.detectionLocation}
                                        </a>   
                                        </span>
                                    </p>
                                    {/* Add Image Display */}
                                    {selectedNotification.carImage && (
                                        <div className="detail-item">
                                            <span className="detail-label">Car Image:</span>
                                            <img
                                                src={selectedNotification.carImage}
                                                alt="Detected Car"
                                                className="car-image"
                                                onError={(e) => e.target.src = 'http://localhost:8000/detections/placeholder.jpg'} // Fallback image
                                            />
                                        </div>
                                    )}
                                     {/* Add "View All Records" Button */}
                                     <div className="view-all-records-button-container">
                                        <Link
                                            to={`/report/${selectedNotification.reportId}`}
                                            className="view-all-records-button"
                                        >
                                            View All Records
                                        </Link>
                                    </div>
                                </div>
                            ) : (
                                <p>Select a notification from the list to view details.</p>
                            )}
                        </div>

                        <div className="notifications-right-panel">
                            {notifications.map((notification) => (
                                <div
                                    key={notification.recordId}
                                    className={`notification-card ${
                                        selectedNotification && selectedNotification.recordId === notification.recordId
                                            ? "selected"
                                            : ""
                                    } ${!notification.seen ? "unread" : ""}`}
                                    onClick={() => handleNotificationClick(notification)}
                                >
                                    <p>
                                        <strong>Notification:</strong> {getNotificationIndex(notification)}
                                    </p>
                        
                                    <p>
                                       {notification.message}
                                    </p>
                                    <button
                                        className="ok-button"
                                        onClick={(e) => {
                                            e.stopPropagation();
                                            handleRemoveNotification(notification.recordId);
                                        }}
                                    >
                                        OK
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="no-notifications">No notifications yet</div>
                )}
            </div>
        </div>
    );
};

export default Notifications;