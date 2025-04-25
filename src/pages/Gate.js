import { Link } from "react-router-dom";
import React from "react";
import "./Gate.css";
import logo from "../assets/logo.png"; // Import the logo image
import car from "../assets/car.png"; // Import the logo image


const Gate = () => {
  return (
    <div className="home-container">
      <nav className="navbar">
        <div className="logo">
          <img src={logo} alt="Carify Logo" className="logo-image" />
        </div>
        <div className="nav-buttons">

          <Link to="/login"> 
            <button className="login-btn">Log In</button>
          </Link>

          <Link to="/signup">
            <button className="signup-btn">Sign Up</button>
          </Link>

        </div>
      </nav>
      <header className="hero-section">
        <div className="text-content">
          <h1>Carify System</h1>
          <p>
            Carify uses AI to detect stolen vehicles by analyzing camera data.
            Enter plate number, model, and color to get immediate alerts when the
            vehicle is detected, enhancing security and speeding up recovery.
          </p>
        </div>
        <div className="image-content">
          <img src={car} alt="car detection" className="car-image" />
        </div>
      </header>
    </div>
  );
};

export default Gate;
