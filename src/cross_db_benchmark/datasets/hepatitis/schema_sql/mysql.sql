-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: Hepatitis_std
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
-- Table structure for table `Bio`
--

DROP TABLE IF EXISTS `Bio`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Bio` (
  `fibros` varchar(45) NOT NULL,
  `activity` varchar(45) NOT NULL,
  `b_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`b_id`),
  KEY `Hepatitis_fibros` (`fibros`),
  KEY `Hepatitis_activity` (`activity`)
) ENGINE=InnoDB AUTO_INCREMENT=33 DEFAULT CHARSET=latin1 AVG_ROW_LENGTH=512;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `dispat`
--

DROP TABLE IF EXISTS `dispat`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `dispat` (
  `m_id` int(10) unsigned NOT NULL DEFAULT 0,
  `sex` varchar(45) DEFAULT NULL COMMENT 'Type: Categorical; Aggr: COUNT',
  `age` varchar(45) DEFAULT NULL,
  `Type` varchar(45) DEFAULT NULL COMMENT 'Type: Categorical; Aggr: B, C',
  PRIMARY KEY (`m_id`),
  KEY `dispat_sex` (`sex`),
  KEY `dispat_age` (`age`),
  KEY `dispat_Type` (`Type`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AVG_ROW_LENGTH=85;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `indis`
--

DROP TABLE IF EXISTS `indis`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `indis` (
  `got` varchar(10) DEFAULT NULL,
  `gpt` varchar(10) DEFAULT NULL,
  `alb` varchar(45) DEFAULT NULL,
  `tbil` varchar(45) DEFAULT NULL,
  `dbil` varchar(45) DEFAULT NULL,
  `che` varchar(45) DEFAULT NULL,
  `ttt` varchar(45) DEFAULT NULL,
  `ztt` varchar(45) DEFAULT NULL,
  `tcho` varchar(45) DEFAULT NULL,
  `tp` varchar(45) DEFAULT NULL,
  `in_id` int(10) unsigned NOT NULL AUTO_INCREMENT,
  PRIMARY KEY (`in_id`),
  KEY `indis_got` (`got`),
  KEY `indis_gpt` (`gpt`),
  KEY `indis_alb` (`alb`),
  KEY `indis_tbil` (`tbil`),
  KEY `indis_dbil` (`dbil`),
  KEY `indis_che` (`che`),
  KEY `indis_ttt` (`ttt`),
  KEY `indis_ztt` (`ztt`),
  KEY `indis_tcho` (`tcho`),
  KEY `indis_tp` (`tp`)
) ENGINE=InnoDB AUTO_INCREMENT=5692 DEFAULT CHARSET=latin1 AVG_ROW_LENGTH=50;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `inf`
--

DROP TABLE IF EXISTS `inf`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `inf` (
  `dur` varchar(45) DEFAULT NULL,
  `a_id` int(10) unsigned NOT NULL DEFAULT 0,
  PRIMARY KEY (`a_id`),
  KEY `inf_dur` (`dur`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AVG_ROW_LENGTH=83;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `rel11`
--

DROP TABLE IF EXISTS `rel11`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `rel11` (
  `b_id` int(10) unsigned NOT NULL DEFAULT 0,
  `m_id` int(10) unsigned NOT NULL DEFAULT 0,
  PRIMARY KEY (`b_id`,`m_id`),
  KEY `FK_rel11_2` (`m_id`),
  CONSTRAINT `FK_rel11_1` FOREIGN KEY (`b_id`) REFERENCES `Bio` (`b_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK_rel11_2` FOREIGN KEY (`m_id`) REFERENCES `dispat` (`m_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AVG_ROW_LENGTH=63;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `rel12`
--

DROP TABLE IF EXISTS `rel12`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `rel12` (
  `in_id` int(10) unsigned NOT NULL DEFAULT 0,
  `m_id` int(10) unsigned NOT NULL DEFAULT 0,
  PRIMARY KEY (`in_id`,`m_id`),
  KEY `FK_rel12_1` (`m_id`),
  CONSTRAINT `FK_rel12_1` FOREIGN KEY (`m_id`) REFERENCES `dispat` (`m_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK_rel12_2` FOREIGN KEY (`in_id`) REFERENCES `indis` (`in_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AVG_ROW_LENGTH=42;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `rel13`
--

DROP TABLE IF EXISTS `rel13`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `rel13` (
  `a_id` int(10) unsigned NOT NULL DEFAULT 0,
  `m_id` int(10) unsigned NOT NULL DEFAULT 0,
  PRIMARY KEY (`a_id`,`m_id`),
  KEY `FK_rel13_1` (`m_id`),
  CONSTRAINT `FK_rel13_1` FOREIGN KEY (`m_id`) REFERENCES `dispat` (`m_id`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `FK_rel13_2` FOREIGN KEY (`a_id`) REFERENCES `inf` (`a_id`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1 AVG_ROW_LENGTH=83;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-02-22 16:58:03
