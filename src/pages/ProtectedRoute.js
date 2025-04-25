import React, { useEffect, useState } from "react";
import { Navigate, useLocation } from "react-router-dom";
import axios from "axios";

const ProtectedRoute = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(null); // null = loading, true = authenticated, false = not authenticated
  const location = useLocation();

  useEffect(() => {
    console.log("ProtectedRoute: Checking authentication...");

    const checkAuth = async () => {
      try {
        const response = await axios.get("http://localhost:8000/api/me", {
          withCredentials: true, // Include cookies in the request
        });

        console.log("ProtectedRoute: /api/me response:", response.data);

        if (response.status === 200) {
          setIsAuthenticated(true); // User is authenticated
          console.log("ProtectedRoute: User is authenticated");
        }
      } catch (error) {
        console.error("ProtectedRoute: Authentication check failed:", error.response?.data || error.message);
        setIsAuthenticated(false); // User is not authenticated
        console.log("ProtectedRoute: User is not authenticated");
      }
    };

    checkAuth();
  }, []);

  // Show loading state while checking authentication
  if (isAuthenticated === null) {
    console.log("ProtectedRoute: Loading...");
    return <div className="loading">Loading...</div>;
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    console.log("ProtectedRoute: Redirecting to /login...");
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Render the protected content if authenticated
  console.log("ProtectedRoute: Rendering children...");
  return children;
};

export default ProtectedRoute;