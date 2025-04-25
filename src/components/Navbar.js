import React, { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import axios from "axios";
import { UserCircleIcon } from "lucide-react";
import "./Navbar.css";
import logo from "../assets/logo.png";

const Navbar = () => {
  const [user, setUser] = useState(null);
  const [showProfile, setShowProfile] = useState(false);
  const navigate = useNavigate();

  // Fetch user data on component mount
  useEffect(() => {
    const fetchUser = async () => {
      try {
        const response = await axios.get("http://localhost:8000/api/me", {
          withCredentials: true,
        });
        console.log("User data:", response.data); // Log the response
        setUser(response.data);
      } catch (error) {
        console.error("Failed to fetch user data:", error.response?.data || error.message);
        setUser(null);
      }
    };
  
    fetchUser();
  }, []);

  // Handle logout
  const handleLogout = async () => {
    try {
      await axios.post("http://localhost:8000/api/logout", null, {
        withCredentials: true, // Include cookies in the request
      });

      setUser(null); // Clear user data
      navigate("/login"); // Redirect to login page
    } catch (error) {
      console.error("Logout failed:", error.response?.data || error.message);
    }
  };


  // Toggle profile visibility
  const toggleProfile = () => setShowProfile(!showProfile);

  return (
    <nav className="navbar">
      <div className="logo">
        <img src={logo} alt="Carify Logo" className="logo-image" />
      </div>
      <ul className="nav-links">
        <li><Link to="/home">Home</Link></li>
        <li><Link to="/reports">Reports</Link></li>
        <li><Link to="/notifications">Notifications</Link></li>
        <li><Link to="/about">About Us</Link></li>
      </ul>
      
      <div className="user-section">
        <div className="user-icon" onClick={toggleProfile}>
          <UserCircleIcon size={32} />
        </div>
        
        {showProfile && user && (
          <div className="profile-dropdown">
            <div className="profile-header">
              <h3>Welcome, {user.name || 'User'}</h3>
              <p className="user-email">{user.email || ''}</p>
            </div>
            <button 
              onClick={handleLogout}
              className="dropdown-logout"
            >
              Logout
            </button>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;