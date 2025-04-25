import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Notification.css';

const Notification = ({ id, message, onClose }) => {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate('/notifications');
    };

    const handleClose = (e) => {
        e.stopPropagation(); // Prevents navigation when closing
        onClose(id);
    };

    return (
        <div className="notification" onClick={handleClick}>
            <p>{message}</p>
            <button onClick={handleClose}>Close</button>
        </div>
    );
};

export default Notification;