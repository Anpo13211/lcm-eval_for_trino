-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: VisualGenome
-- ------------------------------------------------------
-- Server version	5.5.5-10.3.15-MariaDB-log

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `ATT_CLASSES`
--

DROP TABLE IF EXISTS `ATT_CLASSES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ATT_CLASSES` (
  `ATT_CLASS_ID` int(11) NOT NULL DEFAULT 0,
  `ATT_CLASS` char(50) NOT NULL,
  PRIMARY KEY (`ATT_CLASS_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IMG_OBJ`
--

DROP TABLE IF EXISTS `IMG_OBJ`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `IMG_OBJ` (
  `IMG_ID` bigint(20) NOT NULL DEFAULT 0,
  `OBJ_SAMPLE_ID` int(11) NOT NULL DEFAULT 0,
  `OBJ_CLASS_ID` int(11) DEFAULT NULL,
  `X` int(11) DEFAULT NULL,
  `Y` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `H` int(11) DEFAULT NULL,
  PRIMARY KEY (`IMG_ID`,`OBJ_SAMPLE_ID`),
  KEY `OBJ_CLASS_ID` (`OBJ_CLASS_ID`),
  CONSTRAINT `IMG_OBJ_ibfk_1` FOREIGN KEY (`OBJ_CLASS_ID`) REFERENCES `OBJ_CLASSES` (`OBJ_CLASS_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IMG_OBJ_ATT`
--

DROP TABLE IF EXISTS `IMG_OBJ_ATT`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `IMG_OBJ_ATT` (
  `IMG_ID` bigint(20) NOT NULL DEFAULT 0,
  `ATT_CLASS_ID` int(11) NOT NULL DEFAULT 0,
  `OBJ_SAMPLE_ID` int(11) NOT NULL DEFAULT 0,
  PRIMARY KEY (`IMG_ID`,`ATT_CLASS_ID`,`OBJ_SAMPLE_ID`),
  KEY `ATT_CLASS_ID` (`ATT_CLASS_ID`),
  KEY `IMG_ID` (`IMG_ID`,`OBJ_SAMPLE_ID`),
  CONSTRAINT `IMG_OBJ_ATT_ibfk_1` FOREIGN KEY (`ATT_CLASS_ID`) REFERENCES `ATT_CLASSES` (`ATT_CLASS_ID`),
  CONSTRAINT `IMG_OBJ_ATT_ibfk_2` FOREIGN KEY (`IMG_ID`, `OBJ_SAMPLE_ID`) REFERENCES `IMG_OBJ` (`IMG_ID`, `OBJ_SAMPLE_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `IMG_REL`
--

DROP TABLE IF EXISTS `IMG_REL`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `IMG_REL` (
  `IMG_ID` bigint(20) NOT NULL DEFAULT 0,
  `PRED_CLASS_ID` int(11) NOT NULL DEFAULT 0,
  `OBJ1_SAMPLE_ID` int(11) NOT NULL DEFAULT 0,
  `OBJ2_SAMPLE_ID` int(11) NOT NULL DEFAULT 0,
  PRIMARY KEY (`IMG_ID`,`PRED_CLASS_ID`,`OBJ1_SAMPLE_ID`,`OBJ2_SAMPLE_ID`),
  KEY `PRED_CLASS_ID` (`PRED_CLASS_ID`),
  KEY `IMG_ID` (`IMG_ID`,`OBJ1_SAMPLE_ID`),
  KEY `IMG_ID_2` (`IMG_ID`,`OBJ2_SAMPLE_ID`),
  CONSTRAINT `IMG_REL_ibfk_1` FOREIGN KEY (`PRED_CLASS_ID`) REFERENCES `PRED_CLASSES` (`PRED_CLASS_ID`),
  CONSTRAINT `IMG_REL_ibfk_2` FOREIGN KEY (`IMG_ID`, `OBJ1_SAMPLE_ID`) REFERENCES `IMG_OBJ` (`IMG_ID`, `OBJ_SAMPLE_ID`),
  CONSTRAINT `IMG_REL_ibfk_3` FOREIGN KEY (`IMG_ID`, `OBJ2_SAMPLE_ID`) REFERENCES `IMG_OBJ` (`IMG_ID`, `OBJ_SAMPLE_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `OBJ_CLASSES`
--

DROP TABLE IF EXISTS `OBJ_CLASSES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `OBJ_CLASSES` (
  `OBJ_CLASS_ID` int(11) NOT NULL DEFAULT 0,
  `OBJ_CLASS` char(50) NOT NULL,
  PRIMARY KEY (`OBJ_CLASS_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `PRED_CLASSES`
--

DROP TABLE IF EXISTS `PRED_CLASSES`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `PRED_CLASSES` (
  `PRED_CLASS_ID` int(11) NOT NULL DEFAULT 0,
  `PRED_CLASS` char(100) NOT NULL,
  PRIMARY KEY (`PRED_CLASS_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-02-22 17:04:19
