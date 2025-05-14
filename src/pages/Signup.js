import React, { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import logo from "../assets/logo.png";
import "./Signup.css";

const Signup = () => {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [registrationStatus, setRegistrationStatus] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const [passwordError, setPasswordError] = useState(""); 
  const [confirmPasswordError, setConfirmPasswordError] = useState(""); 
  const [emailError, setEmailError] = useState(""); 
  const navigate = useNavigate();
  
  // Password Policy Regex (allows only @, $, !, _ as special characters)
  const passwordPolicyRegex =
    /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!_])[A-Za-z\d@$!_]{8,}$/;


   // Email Validation Regex
   const emailValidationRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  

  // Function to validate email in real-time
  const validateEmail = (value) => {
    if (!emailValidationRegex.test(value)) {
      setEmailError("Please enter a valid email address.");
    } else {
      setEmailError(""); // Clear error if email is valid
    }
  }; 

  // Function to validate password in real-time
  const validatePassword = (value) => {
    if (!passwordPolicyRegex.test(value)) {
      setPasswordError(
        "Password must be at least 8 characters long and include uppercase, lowercase, numbers, and one of the following special characters:( @, $, !, _ )"
      );
    } else {
      setPasswordError(""); // Clear error if password is valid
    }
  };

  // Function to validate confirm password in real-time
  const validateConfirmPassword = (value) => {
    if (value !== password) {
      setConfirmPasswordError("Passwords do not match.");
    } else {
      setConfirmPasswordError(""); // Clear error if passwords match
    }
  };

  // Function to check if the form is valid before submission
  const isFormValid = () => {
      // Validate email
    if (!emailValidationRegex.test(email)) {
      setEmailError("Please enter a valid email address.");
      return false;
    }



    if (!passwordPolicyRegex.test(password)) {
      setPasswordError(
        "Password must be at least 8 characters long and include uppercase, lowercase, numbers, and special characters (!@#$%^&*)."
      );
      return false;
    }

    if (password !== confirmPassword) {
      setConfirmPasswordError("Passwords do not match.");
      return false;
    }

    return true; // Form is valid
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Reset messages
    setRegistrationStatus(null);
    setErrorMessage("");

    // Validate the form before submission
    if (!isFormValid()) {
      return; // Stop submission if the form is invalid
    }

    try {
      const response = await fetch("http://localhost:8000/api/signup", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include", // Include cookies in the request
        body: JSON.stringify({ name, email, password }),
      });

      const data = await response.json();
      if (!response.ok) {
        setErrorMessage(data.message || "An error occurred during registration");
        return;
      }

      // Success message
      setRegistrationStatus("Signup successful! Redirecting to login...");

      // Redirect to login after 1.5 seconds
      setTimeout(() => {
        navigate("/login");
      }, 1500);
    } catch (error) {
      console.error("Error submitting form:", error);
      setErrorMessage("An error occurred while registering the user.");
    }
  };

  return (
    <div className="signup-container">
      <div className="signup-card">
        <Link to="/" className="logo-link">
          <img src={logo} alt="Carify Logo" className="logo-image" />
        </Link>
        <h2>Sign Up</h2>

        {/* Display success or error messages */}
        {registrationStatus && (
          <div className="message-box success-message-box">
            {registrationStatus}
          </div>
        )}
        {errorMessage && <p className="error-message">{errorMessage}</p>}
        {emailError && <p className="error-message">{emailError}</p>}       
        {passwordError && <p className="error-message">{passwordError}</p>}
        {confirmPasswordError && (
          <p className="error-message">{confirmPasswordError}</p>
        )}

        <form onSubmit={handleSubmit}>
          {/* Name Field */}
          <label htmlFor="name">Name</label>
          <input
            type="text"
            id="name"
            placeholder="Enter your name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            required
          />

          {/* Email Field */}
          <label htmlFor="email">Email</label>
          <input
            type="email"
            id="email"
            placeholder="username@gmail.com"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              validateEmail(e.target.value); // Validate email in real-time
            }}
            required
          />

          {/* Password Field */}
          <label htmlFor="password">Password</label>
          <input
            type="password"
            id="password"
            placeholder="Password"
            value={password}
            onChange={(e) => {
              setPassword(e.target.value);
              validatePassword(e.target.value); // Validate password in real-time
            }}
            required
          />

          {/* Confirm Password Field */}
          <label htmlFor="confirm-password">Confirm Password</label>
          <input
            type="password"
            id="confirm-password"
            placeholder="Confirm Password"
            value={confirmPassword}
            onChange={(e) => {
              setConfirmPassword(e.target.value);
              validateConfirmPassword(e.target.value); // Validate confirm password in real-time
            }}
            required
          />

          {/* Submit Button */}
          <button type="submit" className="signup-button">
            Sign Up
          </button>
        </form>
        {/* Link to Login Page */}
        <p className="login-redirect-text">
          Already have an account?{" "}
          <Link to="/login" className="login-redirect-link">
            Log in here.
          </Link>
        </p>
      </div>
    </div>
  );
};

export default Signup;
