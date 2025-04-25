// NewReport.js
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Navbar from "../components/Navbar";
import "./NewReport.css";


export default function NewReport(){
    const navigate = useNavigate();
    const [formData, setFormData] = useState({
        plateNumber: "",
        model: "",
        color: ""
    });

    const [plateNumberError, setPlateNumberError] = useState(""); // For plate number validation errors
   
    // Regex for validating the plate number
    const plateNumberRegex = /^[0-9]{1,4}[A-Za-z]{3}$/;
     



     // Color options map
  const COLOR_MAP = {
    0: "beige",
    1: "black",
    2: "blue",
    3: "green",
    4: "grey",
    5: "red",
    6: "white",
    7: "yellow",
  };


    // Function to validate plate number in real-time
   const validatePlateNumber = (value) => {
    if (!plateNumberRegex.test(value)) {
      setPlateNumberError(
        "Plate number must be 7 characters maximum: 1-4 numbers followed by exactly 3 letters."
      );
    } else {
      setPlateNumberError(""); // Clear error if plate number is valid
    }
  };
    
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });

    // Validate plate number if the input field is "plateNumber"
    if (name === "plateNumber") {
      validatePlateNumber(value);
    }
  };
    

    const handleSubmit = async (e) => {
        e.preventDefault();  
        
        // Validate plate number before submission
      if (!plateNumberRegex.test(formData.plateNumber)) {
        setPlateNumberError(
           "Plate number must be 7 characters maximum: 1-4 numbers followed by exactly 3 letters."
         );
      return; // Stop submission if the plate number is invalid
      }
      
        try {
          console.log("Submitting report with data:", formData);
      
          const response = await fetch("http://localhost:8000/api/report", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            credentials: "include", // Include cookies in the request
            body: JSON.stringify(formData), // Send formData as-is (year remains a string)
          });
      
          console.log("Response status:", response.status);
      
          if (response.status === 401) {
            alert("Please login to submit reports");
            navigate("/login"); // Redirect to login if not authenticated
            return;
          }
      
          const result = await response.json();
          console.log("Response data:", result);
      
          if (response.ok) {
            alert("Report submitted successfully!");
            setFormData({ plateNumber: "", model: "", color: "" });
            navigate("/reports");
          } else {
            alert(`Error: ${result.message || "A report for this car already exists."}`);
          }
        } catch (error) {
          console.error("Error submitting report:", error);
          alert("Failed to submit report due to a network or server error.");
        }
      };
    
    return (
        <div>
            <Navbar />
            <div className="new-report-container">
                <form className="report-form" onSubmit={handleSubmit}>
                    <h2 className="center-title">New Report</h2>
                    <div className="form-group">
                        <label htmlFor="plateNumber">Plate Number</label>
                        <input
                            type="text"
                            id="plateNumber"
                            name="plateNumber"
                            placeholder="Enter car Plate Number"
                            value={formData.plateNumber}
                            onChange={handleChange}
                            required
                        />
                       {plateNumberError && <p className="error-message">{plateNumberError}</p>}

                    </div>
                    
                    <div className="form-group">
                        <label htmlFor="model">Model</label>
                        <input
                            type="text"
                            id="model"
                            name="model"
                            placeholder="Enter Car Model"
                            value={formData.model}
                            onChange={handleChange}
                            required
                        />
                    </div>
                    
                    <div className="form-group">
                        <label htmlFor="Color">Color</label>
                        <select
                          id="color"
                          name="color"
                          value={formData.color}
                          onChange={handleChange}
                          required
                      >
                         <option value="" disabled>Select a color</option>
                         {Object.entries(COLOR_MAP).map(([key, value]) => (
                           <option key={key} value={value}>
                             {value.charAt(0).toUpperCase() + value.slice(1)} {/* Capitalize first letter */}
                           </option>
                        ))}
                      </select>
                    </div>
                    <button type="submit">Submit</button>
                </form>
            </div>
        </div>
    );
}
