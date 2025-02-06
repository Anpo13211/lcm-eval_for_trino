-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: Walmart
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
-- Table structure for table `key`
--

DROP TABLE IF EXISTS `key`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `key` (
  `store_nbr` int(11) NOT NULL,
  `station_nbr` int(11) DEFAULT NULL,
  PRIMARY KEY (`store_nbr`),
  UNIQUE KEY `key_store_nbr_key` (`store_nbr`),
  KEY `key_station_nbr_fkey` (`station_nbr`),
  CONSTRAINT `key_station_nbr_fkey` FOREIGN KEY (`station_nbr`) REFERENCES `station` (`station_nbr`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `station`
--

DROP TABLE IF EXISTS `station`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `station` (
  `station_nbr` int(11) NOT NULL,
  PRIMARY KEY (`station_nbr`),
  UNIQUE KEY `station_station_nbr_key` (`station_nbr`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `train`
--

DROP TABLE IF EXISTS `train`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `train` (
  `date` date NOT NULL,
  `store_nbr` int(11) NOT NULL,
  `item_nbr` int(11) NOT NULL,
  `units` int(11) DEFAULT NULL,
  PRIMARY KEY (`store_nbr`,`date`,`item_nbr`),
  CONSTRAINT `train_store_nbr_fkey` FOREIGN KEY (`store_nbr`) REFERENCES `key` (`store_nbr`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `weather`
--

DROP TABLE IF EXISTS `weather`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `weather` (
  `station_nbr` int(11) NOT NULL,
  `date` date NOT NULL,
  `tmax` int(11) DEFAULT NULL,
  `tmin` int(11) DEFAULT NULL,
  `tavg` int(11) DEFAULT NULL,
  `depart` int(11) DEFAULT NULL,
  `dewpoint` int(11) DEFAULT NULL,
  `wetbulb` int(11) DEFAULT NULL,
  `heat` int(11) DEFAULT NULL,
  `cool` int(11) DEFAULT NULL,
  `sunrise` time DEFAULT NULL,
  `sunset` time DEFAULT NULL,
  `codesum` varchar(255) DEFAULT NULL,
  `snowfall` float DEFAULT NULL,
  `preciptotal` float DEFAULT NULL,
  `stnpressure` float DEFAULT NULL,
  `sealevel` float DEFAULT NULL,
  `resultspeed` float DEFAULT NULL,
  `resultdir` int(11) DEFAULT NULL,
  `avgspeed` float DEFAULT NULL,
  PRIMARY KEY (`station_nbr`,`date`),
  CONSTRAINT `weather_station_nbr_fkey` FOREIGN KEY (`station_nbr`) REFERENCES `station` (`station_nbr`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2021-02-17 15:07:37
