import React from "react";
import "./About.css";
import Navbar from "../components/Navbar";

const About = () => {
  console.log("Rendering Home component");

  return (
    <div className="home-container">
      <Navbar />
      <main className="hero">
        <h3>About us</h3>
        <div className="about-content-container">
          <p>
            Carify is an AI-powered targeted car detection system, specializing in accurately and effectively monitoring and tracking targeted vehicles. In alignment with Saudi Vision 2030, we are committed to enhancing security and driving digital transformation in vehicle management, through advanced technologies and smart solutions that improve safety and efficiency.
          </p>
        </div>
      </main>
    </div>
  );
};

export default About;