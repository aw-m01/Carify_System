import React from "react";
import { Link } from "react-router-dom";
import "./Home.css";
import Navbar from "../components/Navbar";

const Home = () => {
   
  console.log("Rendering Home component");


  return (
    <div className="home-container">
     <Navbar />
      <main className="hero">
        <h1>Welcome to Carify System</h1>
        <div className="button-container">
          <Link to="/new-report">
            <button className="new-report-button">New Report</button>
          </Link>
        </div>
      </main>
    </div>
  );
};

export default Home;
