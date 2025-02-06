-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: imdb_MovieLens
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
-- Table structure for table `actors`
--

DROP TABLE IF EXISTS `actors`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `actors` (
  `actorid` mediumint(8) unsigned NOT NULL,
  `a_gender` enum('M','F') NOT NULL,
  `a_quality` int(2) NOT NULL,
  PRIMARY KEY (`actorid`),
  KEY `a_gender` (`a_gender`),
  KEY `a_quality` (`a_quality`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `directors`
--

DROP TABLE IF EXISTS `directors`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `directors` (
  `directorid` mediumint(8) unsigned NOT NULL,
  `d_quality` int(2) NOT NULL,
  `avg_revenue` int(11) NOT NULL,
  PRIMARY KEY (`directorid`),
  KEY `d_quality` (`d_quality`),
  KEY `avg_revenue` (`avg_revenue`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1 COMMENT='version without Gender information of Directors.';
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `movies`
--

DROP TABLE IF EXISTS `movies`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `movies` (
  `movieid` mediumint(8) unsigned NOT NULL DEFAULT 0,
  `year` int(11) NOT NULL,
  `isEnglish` enum('T','F') NOT NULL,
  `country` varchar(50) NOT NULL,
  `runningtime` int(11) NOT NULL,
  PRIMARY KEY (`movieid`),
  KEY `year` (`year`),
  KEY `isEnglish` (`isEnglish`),
  KEY `country` (`country`),
  KEY `runningtime` (`runningtime`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `movies2actors`
--

DROP TABLE IF EXISTS `movies2actors`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `movies2actors` (
  `movieid` mediumint(8) unsigned NOT NULL,
  `actorid` mediumint(8) unsigned NOT NULL,
  `cast_num` int(11) NOT NULL,
  PRIMARY KEY (`movieid`,`actorid`),
  KEY `fk_m2a_mid_idx` (`movieid`),
  KEY `fk_m2a_aid_idx` (`actorid`),
  KEY `cast_num` (`cast_num`),
  CONSTRAINT `fk_m2a_aid` FOREIGN KEY (`actorid`) REFERENCES `actors` (`actorid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_m2a_mid` FOREIGN KEY (`movieid`) REFERENCES `movies` (`movieid`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `movies2directors`
--

DROP TABLE IF EXISTS `movies2directors`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `movies2directors` (
  `movieid` mediumint(8) unsigned NOT NULL,
  `directorid` mediumint(8) unsigned NOT NULL,
  `genre` varchar(15) NOT NULL,
  PRIMARY KEY (`movieid`,`directorid`),
  KEY `fk_m2d_mid_idx` (`movieid`),
  KEY `fk_m2d_did_idx` (`directorid`),
  KEY `genre` (`genre`),
  CONSTRAINT `fk_m2d_did` FOREIGN KEY (`directorid`) REFERENCES `directors` (`directorid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_m2d_mid` FOREIGN KEY (`movieid`) REFERENCES `movies` (`movieid`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `u2base`
--

DROP TABLE IF EXISTS `u2base`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `u2base` (
  `userid` int(11) NOT NULL DEFAULT 0,
  `movieid` mediumint(8) unsigned NOT NULL,
  `rating` varchar(45) NOT NULL,
  PRIMARY KEY (`userid`,`movieid`),
  KEY `fk_u2base_uid_idx` (`userid`),
  KEY `fk_u2base_mid_idx` (`movieid`),
  KEY `rating` (`rating`),
  CONSTRAINT `fk_u2base_mid` FOREIGN KEY (`movieid`) REFERENCES `movies` (`movieid`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fk_u2base_uid` FOREIGN KEY (`userid`) REFERENCES `users` (`userid`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=latin1;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `users`
--

DROP TABLE IF EXISTS `users`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `users` (
  `userid` int(11) NOT NULL DEFAULT 0,
  `age` varchar(5) NOT NULL,
  `u_gender` varchar(5) NOT NULL,
  `occupation` varchar(45) NOT NULL,
  PRIMARY KEY (`userid`),
  KEY `index` (`age`,`u_gender`,`occupation`),
  KEY `age` (`age`),
  KEY `u_gender` (`u_gender`),
  KEY `occupation` (`occupation`)
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

-- Dump completed on 2021-02-22 16:44:11
