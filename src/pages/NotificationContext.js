import React, { createContext, useContext, useState, useEffect } from 'react';
import Notification from './Notification'; // We'll create this next

const NotificationContext = createContext();

export const useNotifications = () => useContext(NotificationContext);

export const NotificationProvider = ({ children }) => {
    const [notifications, setNotifications] = useState([]);

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws/notifications');
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            setNotifications(prev => [...prev, { id: Date.now(), message: data.message }]);
        };
        return () => {
            ws.close();
        };
    }, []);

    const removeNotification = (id) => {
        setNotifications(prev => prev.filter(notif => notif.id !== id));
    };

    return (
        <NotificationContext.Provider value={{ notifications, removeNotification }}>
            {children}
            <div className="notifications-container">
                {notifications.map(notif => (
                    <Notification 
                        key={notif.id} 
                        id={notif.id} 
                        message={notif.message} 
                        onClose={removeNotification} 
                    />
                ))}
            </div>
        </NotificationContext.Provider>
    );
};