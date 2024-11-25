import express from "express";
import mongoose from "mongoose";
import * as dotenv from "dotenv";
import authRoutes from "./routes/auth.js";
import cors from "cors";

const PORT = 8080; 

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());


app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).json({ error: "Something went wrong!" });
});


app.get("/", (req, res) => {
  res.json({ message: "API is running" });
});

app.use("/api/auth", authRoutes);


const connectToMongoDB = async () => {
    try {
        const mongoURI = process.env.MONGODB_URI;
        if (!mongoURI) {
            throw new Error('MONGODB_URI is not defined');
        }
        await mongoose.connect(mongoURI, {
            serverSelectionTimeoutMS: 5000
        });
        console.log('Connected to MongoDB');
    } catch (error) {
        console.error('MongoDB connection error:', error);
        process.exit(1);  
    }
};


const startServer = async () => {
  await connectToMongoDB();

  
  app.listen(PORT, '0.0.0.0', () => {
    
    console.log(`Server is running on port ${PORT}`);
  });
};

startServer();