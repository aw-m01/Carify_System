import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import axios from 'axios';
import './Login.css';
import logo from '../assets/logo.png';

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState(''); 
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(''); // Clear previous errors
    setSuccessMessage(''); // Clear previous success messages
    try {
      const response = await axios.post(
        'http://localhost:8000/api/login', 
        { email, password },
        { headers: { 'Content-Type': 'application/json' },
          withCredentials: true,      
        }
      );
      
       // Check if login was successful
       if (response.status === 200) {
        setSuccessMessage('Login successful! Redirecting to home...');
        setTimeout(() => {
          navigate('/home'); // Redirect after showing success message
        }, 1500); // Delay for 1.5 seconds to show the success message
      } else {
        setError('Login failed: Unexpected response from server');
      }
    } catch (err) {
      console.error('Login Error:', err.response?.data || err.message);
      setError(err.response?.data?.message || 'Invalid email or password. Please try again.');
    }
  };

  return (
    <div className="login-container">
      <div className="login-card">
      <Link to="/" className="logo-link">
          <img src={logo} alt="Carify Logo" className="logo-image" />
        </Link>
        <h2>Login</h2>
         {/* Display Success Message */}
        {successMessage && (
          <div className="success-message-box">
            <p>{successMessage}</p>
          </div>
        )}


         {/* Display Error Message */}
         {error && <p className="error-message">{error}</p>}
       
        <form onSubmit={handleSubmit}>
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            placeholder="username@gmail.com"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <button type="submit" className="login-button">Log In</button>
          <p className="auth-redirect-text">
            Don't have an account? <a href="/signup" className="auth-redirect-link">Sign Up</a>
          </p>
        </form>
      </div>
    </div>
  );
};

export default Login;
