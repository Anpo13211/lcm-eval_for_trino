-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: Carcinogenesis
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
-- Table structure for table `atom`
--

DROP TABLE IF EXISTS `atom`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `atom` (
  `atomid` char(100) NOT NULL,
  `drug` char(100) DEFAULT NULL,
  `atomtype` char(100) DEFAULT NULL,
  `charge` char(100) DEFAULT NULL,
  `name` char(2) DEFAULT NULL,
  PRIMARY KEY (`atomid`),
  KEY `atom_drug` (`drug`) USING BTREE,
  CONSTRAINT `atom_ibfk_1` FOREIGN KEY (`drug`) REFERENCES `canc` (`drug_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `canc`
--

DROP TABLE IF EXISTS `canc`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `canc` (
  `drug_id` char(100) NOT NULL,
  `class` char(1) DEFAULT NULL,
  PRIMARY KEY (`drug_id`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sbond_1`
--

DROP TABLE IF EXISTS `sbond_1`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sbond_1` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `drug` char(100) DEFAULT NULL,
  `atomid` char(100) DEFAULT NULL,
  `atomid_2` char(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sbond_1_drug` (`drug`) USING BTREE,
  KEY `sbond_1_atomid` (`atomid`) USING BTREE,
  KEY `sbond_1_atomid_2` (`atomid_2`) USING BTREE,
  CONSTRAINT `sbond_1_ibfk_1` FOREIGN KEY (`drug`) REFERENCES `canc` (`drug_id`),
  CONSTRAINT `sbond_1_ibfk_2` FOREIGN KEY (`atomid`) REFERENCES `atom` (`atomid`),
  CONSTRAINT `sbond_1_ibfk_3` FOREIGN KEY (`atomid_2`) REFERENCES `atom` (`atomid`)
) ENGINE=InnoDB AUTO_INCREMENT=13563 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sbond_2`
--

DROP TABLE IF EXISTS `sbond_2`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sbond_2` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `drug` char(100) DEFAULT NULL,
  `atomid` char(100) DEFAULT NULL,
  `atomid_2` char(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sbond_2_drug` (`drug`) USING BTREE,
  KEY `sbond_2_atomid` (`atomid`) USING BTREE,
  KEY `sbond_2_atomid_2` (`atomid_2`) USING BTREE,
  CONSTRAINT `sbond_2_ibfk_1` FOREIGN KEY (`drug`) REFERENCES `canc` (`drug_id`),
  CONSTRAINT `sbond_2_ibfk_2` FOREIGN KEY (`atomid`) REFERENCES `atom` (`atomid`),
  CONSTRAINT `sbond_2_ibfk_3` FOREIGN KEY (`atomid_2`) REFERENCES `atom` (`atomid`)
) ENGINE=InnoDB AUTO_INCREMENT=927 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sbond_3`
--

DROP TABLE IF EXISTS `sbond_3`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sbond_3` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `drug` char(100) DEFAULT NULL,
  `atomid` char(100) DEFAULT NULL,
  `atomid_2` char(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sbond_3_drug` (`drug`) USING BTREE,
  KEY `sbond_3_atomid` (`atomid`) USING BTREE,
  KEY `sbond_3_atomid_2` (`atomid_2`) USING BTREE,
  CONSTRAINT `sbond_3_ibfk_1` FOREIGN KEY (`drug`) REFERENCES `canc` (`drug_id`),
  CONSTRAINT `sbond_3_ibfk_2` FOREIGN KEY (`atomid`) REFERENCES `atom` (`atomid`),
  CONSTRAINT `sbond_3_ibfk_3` FOREIGN KEY (`atomid_2`) REFERENCES `atom` (`atomid`)
) ENGINE=InnoDB AUTO_INCREMENT=13 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `sbond_7`
--

DROP TABLE IF EXISTS `sbond_7`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `sbond_7` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `drug` char(100) DEFAULT NULL,
  `atomid` char(100) DEFAULT NULL,
  `atomid_2` char(100) DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `sbond_7_drug` (`drug`) USING BTREE,
  KEY `sbond_7_atomid` (`atomid`) USING BTREE,
  KEY `sbond_7_atomid_2` (`atomid_2`) USING BTREE,
  CONSTRAINT `sbond_7_ibfk_1` FOREIGN KEY (`drug`) REFERENCES `canc` (`drug_id`),
  CONSTRAINT `sbond_7_ibfk_2` FOREIGN KEY (`atomid`) REFERENCES `atom` (`atomid`),
  CONSTRAINT `sbond_7_ibfk_3` FOREIGN KEY (`atomid_2`) REFERENCES `atom` (`atomid`)
) ENGINE=InnoDB AUTO_INCREMENT=4135 DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-02-22 17:31:38
