-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: lahman_2014
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
-- Table structure for table `allstarfull`
--

DROP TABLE IF EXISTS `allstarfull`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `allstarfull` (
  `playerID` varchar(9) NOT NULL,
  `yearID` int(11) NOT NULL,
  `gameNum` int(11) NOT NULL,
  `gameID` varchar(12) DEFAULT NULL,
  `teamID` varchar(3) DEFAULT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `GP` int(11) DEFAULT NULL,
  `startingPos` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`yearID`,`gameNum`),
  KEY `allstarfull_teamID` (`teamID`),
  CONSTRAINT `allstarfull_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `appearances`
--

DROP TABLE IF EXISTS `appearances`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `appearances` (
  `yearID` int(11) NOT NULL,
  `teamID` varchar(3) NOT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `playerID` varchar(9) NOT NULL,
  `G_all` int(11) DEFAULT NULL,
  `G_batting` int(11) DEFAULT NULL,
  `G_defense` int(11) DEFAULT NULL,
  `G_p` int(11) DEFAULT NULL,
  `G_c` int(11) DEFAULT NULL,
  `G_1b` int(11) DEFAULT NULL,
  `G_2b` int(11) DEFAULT NULL,
  `G_3b` int(11) DEFAULT NULL,
  `G_ss` int(11) DEFAULT NULL,
  `G_lf` int(11) DEFAULT NULL,
  `G_cf` int(11) DEFAULT NULL,
  `G_rf` int(11) DEFAULT NULL,
  `G_of` int(11) DEFAULT NULL,
  `G_dh` int(11) DEFAULT NULL,
  `G_ph` int(11) DEFAULT NULL,
  `G_pr` int(11) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`teamID`,`playerID`),
  KEY `appearances_teamID` (`teamID`),
  KEY `appearances_playerID` (`playerID`),
  KEY `appearances_playerID_2` (`playerID`),
  CONSTRAINT `appearances_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `awardsmanagers`
--

DROP TABLE IF EXISTS `awardsmanagers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `awardsmanagers` (
  `managerID` varchar(10) NOT NULL,
  `awardID` varchar(25) NOT NULL,
  `yearID` int(11) NOT NULL,
  `lgID` varchar(2) NOT NULL,
  `tie` varchar(1) DEFAULT NULL,
  `notes` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`awardID`,`lgID`,`managerID`),
  KEY `awardsmanagers_managerID` (`managerID`),
  CONSTRAINT `awardsmanagers_ibfk_1` FOREIGN KEY (`managerID`) REFERENCES `managers` (`managerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `awardsplayers`
--

DROP TABLE IF EXISTS `awardsplayers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `awardsplayers` (
  `playerID` varchar(9) NOT NULL,
  `awardID` varchar(255) NOT NULL,
  `yearID` int(11) NOT NULL,
  `lgID` varchar(2) NOT NULL,
  `tie` varchar(1) DEFAULT NULL,
  `notes` varchar(100) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`awardID`,`lgID`,`playerID`),
  KEY `awardsplayers_playerID` (`playerID`),
  CONSTRAINT `awardsplayers_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `awardssharemanagers`
--

DROP TABLE IF EXISTS `awardssharemanagers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `awardssharemanagers` (
  `awardID` varchar(25) NOT NULL,
  `yearID` int(11) NOT NULL,
  `lgID` varchar(2) NOT NULL,
  `managerID` varchar(10) NOT NULL,
  `pointsWon` int(11) DEFAULT NULL,
  `pointsMax` int(11) DEFAULT NULL,
  `votesFirst` int(11) DEFAULT NULL,
  PRIMARY KEY (`awardID`,`yearID`,`lgID`,`managerID`),
  KEY `awardssharemanagers_managerID` (`managerID`),
  CONSTRAINT `awardssharemanagers_ibfk_1` FOREIGN KEY (`managerID`) REFERENCES `managers` (`managerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `awardsshareplayers`
--

DROP TABLE IF EXISTS `awardsshareplayers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `awardsshareplayers` (
  `awardID` varchar(25) NOT NULL,
  `yearID` int(11) NOT NULL,
  `lgID` varchar(2) NOT NULL,
  `playerID` varchar(9) NOT NULL,
  `pointsWon` int(11) DEFAULT NULL,
  `pointsMax` int(11) DEFAULT NULL,
  `votesFirst` int(11) DEFAULT NULL,
  PRIMARY KEY (`awardID`,`yearID`,`lgID`,`playerID`),
  KEY `awardsshareplayers_playerID` (`playerID`),
  KEY `awardsshareplayers_playerID_2` (`playerID`),
  CONSTRAINT `awardsshareplayers_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `batting`
--

DROP TABLE IF EXISTS `batting`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `batting` (
  `playerID` varchar(9) NOT NULL,
  `yearID` int(11) NOT NULL,
  `stint` int(11) NOT NULL,
  `teamID` varchar(3) DEFAULT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `G_batting` int(11) DEFAULT NULL,
  `AB` int(11) DEFAULT NULL,
  `R` int(11) DEFAULT NULL,
  `H` int(11) DEFAULT NULL,
  `2B` int(11) DEFAULT NULL,
  `3B` int(11) DEFAULT NULL,
  `HR` int(11) DEFAULT NULL,
  `RBI` int(11) DEFAULT NULL,
  `SB` int(11) DEFAULT NULL,
  `CS` int(11) DEFAULT NULL,
  `BB` int(11) DEFAULT NULL,
  `SO` int(11) DEFAULT NULL,
  `IBB` int(11) DEFAULT NULL,
  `HBP` int(11) DEFAULT NULL,
  `SH` int(11) DEFAULT NULL,
  `SF` int(11) DEFAULT NULL,
  `GIDP` int(11) DEFAULT NULL,
  `G_old` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`yearID`,`stint`),
  KEY `batting_teamID` (`teamID`),
  CONSTRAINT `batting_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `batting_ibfk_2` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `battingpost`
--

DROP TABLE IF EXISTS `battingpost`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `battingpost` (
  `yearID` int(11) NOT NULL,
  `round` varchar(10) NOT NULL,
  `playerID` varchar(9) NOT NULL,
  `teamID` varchar(3) DEFAULT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `AB` int(11) DEFAULT NULL,
  `R` int(11) DEFAULT NULL,
  `H` int(11) DEFAULT NULL,
  `2B` int(11) DEFAULT NULL,
  `3B` int(11) DEFAULT NULL,
  `HR` int(11) DEFAULT NULL,
  `RBI` int(11) DEFAULT NULL,
  `SB` int(11) DEFAULT NULL,
  `CS` int(11) DEFAULT NULL,
  `BB` int(11) DEFAULT NULL,
  `SO` int(11) DEFAULT NULL,
  `IBB` int(11) DEFAULT NULL,
  `HBP` int(11) DEFAULT NULL,
  `SH` int(11) DEFAULT NULL,
  `SF` int(11) DEFAULT NULL,
  `GIDP` int(11) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`round`,`playerID`),
  KEY `battingpost_teamID` (`teamID`),
  KEY `battingpost_playerID` (`playerID`),
  KEY `battingpost_playerID_2` (`playerID`),
  CONSTRAINT `battingpost_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `battingpost_ibfk_2` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `els_teamnames`
--

DROP TABLE IF EXISTS `els_teamnames`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `els_teamnames` (
  `id` int(11) NOT NULL,
  `lgid` varchar(2) NOT NULL,
  `teamid` varchar(3) NOT NULL,
  `franchid` varchar(3) DEFAULT NULL,
  `name` varchar(50) DEFAULT NULL,
  `park` varchar(255) DEFAULT NULL,
  KEY `els_teamnames_teamid` (`teamid`),
  KEY `els_teamnames_franchid` (`franchid`),
  CONSTRAINT `els_teamnames_ibfk_1` FOREIGN KEY (`teamid`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `els_teamnames_ibfk_2` FOREIGN KEY (`franchid`) REFERENCES `teamsfranchises` (`franchID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `fielding`
--

DROP TABLE IF EXISTS `fielding`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `fielding` (
  `playerID` varchar(9) NOT NULL,
  `yearID` int(11) NOT NULL,
  `stint` int(11) NOT NULL,
  `teamID` varchar(3) DEFAULT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `POS` varchar(2) NOT NULL,
  `G` int(11) DEFAULT NULL,
  `GS` int(11) DEFAULT NULL,
  `InnOuts` int(11) DEFAULT NULL,
  `PO` int(11) DEFAULT NULL,
  `A` int(11) DEFAULT NULL,
  `E` int(11) DEFAULT NULL,
  `DP` int(11) DEFAULT NULL,
  `PB` int(11) DEFAULT NULL,
  `WP` int(11) DEFAULT NULL,
  `SB` int(11) DEFAULT NULL,
  `CS` int(11) DEFAULT NULL,
  `ZR` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`yearID`,`stint`,`POS`),
  KEY `fielding_teamID` (`teamID`),
  CONSTRAINT `fielding_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fielding_ibfk_2` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `fieldingof`
--

DROP TABLE IF EXISTS `fieldingof`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `fieldingof` (
  `playerID` varchar(9) NOT NULL,
  `yearID` int(11) NOT NULL,
  `stint` int(11) NOT NULL,
  `Glf` int(11) DEFAULT NULL,
  `Gcf` int(11) DEFAULT NULL,
  `Grf` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`yearID`,`stint`),
  CONSTRAINT `fieldingof_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `fieldingpost`
--

DROP TABLE IF EXISTS `fieldingpost`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `fieldingpost` (
  `playerID` varchar(9) NOT NULL,
  `yearID` int(11) NOT NULL,
  `teamID` varchar(3) DEFAULT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `round` varchar(10) NOT NULL,
  `POS` varchar(2) NOT NULL,
  `G` int(11) DEFAULT NULL,
  `GS` int(11) DEFAULT NULL,
  `InnOuts` int(11) DEFAULT NULL,
  `PO` int(11) DEFAULT NULL,
  `A` int(11) DEFAULT NULL,
  `E` int(11) DEFAULT NULL,
  `DP` int(11) DEFAULT NULL,
  `TP` int(11) DEFAULT NULL,
  `PB` int(11) DEFAULT NULL,
  `SB` int(11) DEFAULT NULL,
  `CS` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`yearID`,`round`,`POS`),
  KEY `fieldingpost_teamID` (`teamID`),
  CONSTRAINT `fieldingpost_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `fieldingpost_ibfk_2` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `halloffame`
--

DROP TABLE IF EXISTS `halloffame`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `halloffame` (
  `hofID` varchar(10) NOT NULL,
  `yearID` int(11) NOT NULL,
  `votedBy` varchar(64) DEFAULT NULL,
  `ballots` int(11) DEFAULT NULL,
  `needed` int(11) DEFAULT NULL,
  `votes` int(11) DEFAULT NULL,
  `inducted` varchar(1) DEFAULT NULL,
  `category` varchar(20) DEFAULT NULL,
  PRIMARY KEY (`hofID`,`yearID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `managers`
--

DROP TABLE IF EXISTS `managers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `managers` (
  `managerID` varchar(10) DEFAULT NULL,
  `yearID` int(11) NOT NULL,
  `teamID` varchar(3) NOT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `inseason` int(11) NOT NULL,
  `G` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `rank` int(11) DEFAULT NULL,
  `plyrMgr` varchar(1) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`teamID`,`inseason`),
  KEY `managers_teamID` (`teamID`),
  KEY `managers_managerID` (`managerID`),
  CONSTRAINT `managers_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `managershalf`
--

DROP TABLE IF EXISTS `managershalf`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `managershalf` (
  `managerID` varchar(10) NOT NULL,
  `yearID` int(11) NOT NULL,
  `teamID` varchar(3) NOT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `inseason` int(11) DEFAULT NULL,
  `half` int(11) NOT NULL,
  `G` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `rank` int(11) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`teamID`,`managerID`,`half`),
  KEY `managershalf_teamID` (`teamID`),
  KEY `managershalf_managerID` (`managerID`),
  CONSTRAINT `managershalf_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `managershalf_ibfk_2` FOREIGN KEY (`managerID`) REFERENCES `managers` (`managerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pitching`
--

DROP TABLE IF EXISTS `pitching`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pitching` (
  `playerID` varchar(9) NOT NULL,
  `yearID` int(11) NOT NULL,
  `stint` int(11) NOT NULL,
  `teamID` varchar(3) DEFAULT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `GS` int(11) DEFAULT NULL,
  `CG` int(11) DEFAULT NULL,
  `SHO` int(11) DEFAULT NULL,
  `SV` int(11) DEFAULT NULL,
  `IPouts` int(11) DEFAULT NULL,
  `H` int(11) DEFAULT NULL,
  `ER` int(11) DEFAULT NULL,
  `HR` int(11) DEFAULT NULL,
  `BB` int(11) DEFAULT NULL,
  `SO` int(11) DEFAULT NULL,
  `BAOpp` int(11) DEFAULT NULL,
  `ERA` int(11) DEFAULT NULL,
  `IBB` int(11) DEFAULT NULL,
  `WP` int(11) DEFAULT NULL,
  `HBP` int(11) DEFAULT NULL,
  `BK` int(11) DEFAULT NULL,
  `BFP` int(11) DEFAULT NULL,
  `GF` int(11) DEFAULT NULL,
  `R` int(11) DEFAULT NULL,
  `SH` int(11) DEFAULT NULL,
  `SF` int(11) DEFAULT NULL,
  `GIDP` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`yearID`,`stint`),
  KEY `pitching_teamID` (`teamID`),
  CONSTRAINT `pitching_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `pitching_ibfk_2` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `pitchingpost`
--

DROP TABLE IF EXISTS `pitchingpost`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pitchingpost` (
  `playerID` varchar(9) NOT NULL,
  `yearID` int(11) NOT NULL,
  `round` varchar(10) NOT NULL,
  `teamID` varchar(3) DEFAULT NULL,
  `lgID` varchar(2) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `GS` int(11) DEFAULT NULL,
  `CG` int(11) DEFAULT NULL,
  `SHO` int(11) DEFAULT NULL,
  `SV` int(11) DEFAULT NULL,
  `IPouts` int(11) DEFAULT NULL,
  `H` int(11) DEFAULT NULL,
  `ER` int(11) DEFAULT NULL,
  `HR` int(11) DEFAULT NULL,
  `BB` int(11) DEFAULT NULL,
  `SO` int(11) DEFAULT NULL,
  `BAOpp` int(11) DEFAULT NULL,
  `ERA` int(11) DEFAULT NULL,
  `IBB` int(11) DEFAULT NULL,
  `WP` int(11) DEFAULT NULL,
  `HBP` int(11) DEFAULT NULL,
  `BK` int(11) DEFAULT NULL,
  `BFP` int(11) DEFAULT NULL,
  `GF` int(11) DEFAULT NULL,
  `R` int(11) DEFAULT NULL,
  `SH` int(11) DEFAULT NULL,
  `SF` int(11) DEFAULT NULL,
  `GIDP` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`yearID`,`round`),
  KEY `pitchingpost_teamID` (`teamID`),
  CONSTRAINT `pitchingpost_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `pitchingpost_ibfk_2` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `players`
--

DROP TABLE IF EXISTS `players`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `players` (
  `lahmanID` int(11) NOT NULL,
  `playerID` varchar(9) DEFAULT NULL,
  `managerID` varchar(10) DEFAULT NULL,
  `hofID` varchar(10) DEFAULT NULL,
  `birthYear` int(11) DEFAULT NULL,
  `birthMonth` int(11) DEFAULT NULL,
  `birthDay` int(11) DEFAULT NULL,
  `birthCountry` varchar(50) DEFAULT NULL,
  `birthState` varchar(2) DEFAULT NULL,
  `birthCity` varchar(50) DEFAULT NULL,
  `deathYear` int(11) DEFAULT NULL,
  `deathMonth` int(11) DEFAULT NULL,
  `deathDay` int(11) DEFAULT NULL,
  `deathCountry` varchar(50) DEFAULT NULL,
  `deathState` varchar(2) DEFAULT NULL,
  `deathCity` varchar(50) DEFAULT NULL,
  `nameFirst` varchar(50) DEFAULT NULL,
  `nameLast` varchar(50) DEFAULT NULL,
  `nameNote` varchar(255) DEFAULT NULL,
  `nameGiven` varchar(255) DEFAULT NULL,
  `nameNick` varchar(255) DEFAULT NULL,
  `weight` int(11) DEFAULT NULL,
  `height` int(11) DEFAULT NULL,
  `bats` varchar(1) DEFAULT NULL,
  `throws` varchar(1) DEFAULT NULL,
  `debut` varchar(10) DEFAULT NULL,
  `finalGame` varchar(10) DEFAULT NULL,
  `college` varchar(50) DEFAULT NULL,
  `lahman40ID` varchar(9) DEFAULT NULL,
  `lahman45ID` varchar(9) DEFAULT NULL,
  `retroID` varchar(9) DEFAULT NULL,
  `holtzID` varchar(9) DEFAULT NULL,
  `bbrefID` varchar(9) DEFAULT NULL,
  PRIMARY KEY (`lahmanID`),
  KEY `players_playerID` (`playerID`),
  KEY `players_hofID` (`hofID`),
  KEY `players_hofID_2` (`hofID`),
  CONSTRAINT `players_ibfk_1` FOREIGN KEY (`hofID`) REFERENCES `halloffame` (`hofID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `salaries`
--

DROP TABLE IF EXISTS `salaries`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `salaries` (
  `yearID` int(11) NOT NULL,
  `teamID` varchar(3) NOT NULL,
  `lgID` varchar(2) NOT NULL,
  `playerID` varchar(9) NOT NULL,
  `salary` int(11) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`teamID`,`lgID`,`playerID`),
  KEY `salaries_teamID` (`teamID`),
  CONSTRAINT `salaries_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `schools`
--

DROP TABLE IF EXISTS `schools`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `schools` (
  `schoolID` varchar(15) NOT NULL,
  `schoolName` varchar(255) DEFAULT NULL,
  `schoolCity` varchar(55) DEFAULT NULL,
  `schoolState` varchar(55) DEFAULT NULL,
  `schoolNick` varchar(55) DEFAULT NULL,
  PRIMARY KEY (`schoolID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `schoolsplayers`
--

DROP TABLE IF EXISTS `schoolsplayers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `schoolsplayers` (
  `playerID` varchar(9) NOT NULL,
  `schoolID` varchar(15) NOT NULL,
  `yearMin` int(11) DEFAULT NULL,
  `yearMax` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`schoolID`),
  KEY `schoolsplayers_schoolID` (`schoolID`),
  CONSTRAINT `schoolsplayers_ibfk_1` FOREIGN KEY (`schoolID`) REFERENCES `schools` (`schoolID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `schoolsplayers_ibfk_2` FOREIGN KEY (`playerID`) REFERENCES `players` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `seriespost`
--

DROP TABLE IF EXISTS `seriespost`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `seriespost` (
  `yearID` int(11) NOT NULL,
  `round` varchar(5) NOT NULL,
  `teamIDwinner` varchar(3) DEFAULT NULL,
  `lgIDwinner` varchar(2) DEFAULT NULL,
  `teamIDloser` varchar(3) DEFAULT NULL,
  `lgIDloser` varchar(2) DEFAULT NULL,
  `wins` int(11) DEFAULT NULL,
  `losses` int(11) DEFAULT NULL,
  `ties` int(11) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`round`),
  KEY `seriespost_teamIDwinner` (`teamIDwinner`),
  KEY `seriespost_teamIDloser` (`teamIDloser`),
  CONSTRAINT `seriespost_ibfk_1` FOREIGN KEY (`teamIDwinner`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `seriespost_ibfk_2` FOREIGN KEY (`teamIDloser`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `teams`
--

DROP TABLE IF EXISTS `teams`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `teams` (
  `yearID` int(11) NOT NULL,
  `lgID` varchar(2) NOT NULL,
  `teamID` varchar(3) NOT NULL,
  `franchID` varchar(3) DEFAULT NULL,
  `divID` varchar(1) DEFAULT NULL,
  `Rank` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `Ghome` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `DivWin` varchar(1) DEFAULT NULL,
  `WCWin` varchar(1) DEFAULT NULL,
  `LgWin` varchar(1) DEFAULT NULL,
  `WSWin` varchar(1) DEFAULT NULL,
  `R` int(11) DEFAULT NULL,
  `AB` int(11) DEFAULT NULL,
  `H` int(11) DEFAULT NULL,
  `2B` int(11) DEFAULT NULL,
  `3B` int(11) DEFAULT NULL,
  `HR` int(11) DEFAULT NULL,
  `BB` int(11) DEFAULT NULL,
  `SO` int(11) DEFAULT NULL,
  `SB` int(11) DEFAULT NULL,
  `CS` int(11) DEFAULT NULL,
  `HBP` int(11) DEFAULT NULL,
  `SF` int(11) DEFAULT NULL,
  `RA` int(11) DEFAULT NULL,
  `ER` int(11) DEFAULT NULL,
  `ERA` int(11) DEFAULT NULL,
  `CG` int(11) DEFAULT NULL,
  `SHO` int(11) DEFAULT NULL,
  `SV` int(11) DEFAULT NULL,
  `IPouts` int(11) DEFAULT NULL,
  `HA` int(11) DEFAULT NULL,
  `HRA` int(11) DEFAULT NULL,
  `BBA` int(11) DEFAULT NULL,
  `SOA` int(11) DEFAULT NULL,
  `E` int(11) DEFAULT NULL,
  `DP` int(11) DEFAULT NULL,
  `FP` int(11) DEFAULT NULL,
  `name` varchar(50) DEFAULT NULL,
  `park` varchar(255) DEFAULT NULL,
  `attendance` int(11) DEFAULT NULL,
  `BPF` int(11) DEFAULT NULL,
  `PPF` int(11) DEFAULT NULL,
  `teamIDBR` varchar(3) DEFAULT NULL,
  `teamIDlahman45` varchar(3) DEFAULT NULL,
  `teamIDretro` varchar(3) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`lgID`,`teamID`),
  KEY `teams_teamID` (`teamID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `teamsfranchises`
--

DROP TABLE IF EXISTS `teamsfranchises`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `teamsfranchises` (
  `franchID` varchar(3) NOT NULL,
  `franchName` varchar(50) DEFAULT NULL,
  `active` varchar(2) DEFAULT NULL,
  `NAassoc` varchar(3) DEFAULT NULL,
  PRIMARY KEY (`franchID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `teamshalf`
--

DROP TABLE IF EXISTS `teamshalf`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `teamshalf` (
  `yearID` int(11) NOT NULL,
  `lgID` varchar(2) NOT NULL,
  `teamID` varchar(3) NOT NULL,
  `Half` varchar(1) NOT NULL,
  `divID` varchar(1) DEFAULT NULL,
  `DivWin` varchar(1) DEFAULT NULL,
  `Rank` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  PRIMARY KEY (`yearID`,`teamID`,`lgID`,`Half`),
  KEY `teamshalf_teamID` (`teamID`),
  CONSTRAINT `teamshalf_ibfk_1` FOREIGN KEY (`teamID`) REFERENCES `teams` (`teamID`) ON DELETE CASCADE ON UPDATE CASCADE
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

-- Dump completed on 2021-02-22 16:48:07
