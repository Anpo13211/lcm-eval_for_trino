-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: Hockey
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
-- Table structure for table `AwardsCoaches`
--

DROP TABLE IF EXISTS `AwardsCoaches`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `AwardsCoaches` (
  `coachID` varchar(255) DEFAULT NULL,
  `award` varchar(255) DEFAULT NULL,
  `year` int(11) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  KEY `AwardsCoaches_coachID` (`coachID`) USING BTREE,
  CONSTRAINT `AwardsCoaches_ibfk_1` FOREIGN KEY (`coachID`) REFERENCES `Coaches` (`coachID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `AwardsMisc`
--

DROP TABLE IF EXISTS `AwardsMisc`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `AwardsMisc` (
  `name` varchar(255) NOT NULL,
  `ID` varchar(255) DEFAULT NULL,
  `award` varchar(255) DEFAULT NULL,
  `year` int(11) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`name`),
  KEY `ID` (`ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `AwardsPlayers`
--

DROP TABLE IF EXISTS `AwardsPlayers`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `AwardsPlayers` (
  `playerID` varchar(255) NOT NULL,
  `award` varchar(255) NOT NULL,
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  `pos` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`award`,`year`),
  CONSTRAINT `AwardsPlayers_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Coaches`
--

DROP TABLE IF EXISTS `Coaches`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Coaches` (
  `coachID` varchar(255) NOT NULL,
  `year` int(11) NOT NULL,
  `tmID` varchar(255) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `stint` int(11) NOT NULL,
  `notes` varchar(255) DEFAULT NULL,
  `g` int(11) DEFAULT NULL,
  `w` int(11) DEFAULT NULL,
  `l` int(11) DEFAULT NULL,
  `t` int(11) DEFAULT NULL,
  `postg` varchar(255) DEFAULT NULL,
  `postw` varchar(255) DEFAULT NULL,
  `postl` varchar(255) DEFAULT NULL,
  `postt` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`coachID`,`year`,`tmID`,`stint`),
  KEY `Coaches_coachID` (`coachID`) USING BTREE,
  KEY `Coaches_year_tmID` (`year`,`tmID`) USING BTREE,
  CONSTRAINT `Coaches_ibfk_1` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `CombinedShutouts`
--

DROP TABLE IF EXISTS `CombinedShutouts`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `CombinedShutouts` (
  `year` int(11) DEFAULT NULL,
  `month` int(11) DEFAULT NULL,
  `date` int(11) DEFAULT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `oppID` varchar(255) DEFAULT NULL,
  `R/P` varchar(255) DEFAULT NULL,
  `IDgoalie1` varchar(255) DEFAULT NULL,
  `IDgoalie2` varchar(255) DEFAULT NULL,
  KEY `CombinedShutouts_IDgoalie1` (`IDgoalie1`) USING BTREE,
  KEY `CombinedShutouts_IDgoalie2` (`IDgoalie2`) USING BTREE,
  CONSTRAINT `CombinedShutouts_ibfk_1` FOREIGN KEY (`IDgoalie1`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `CombinedShutouts_ibfk_2` FOREIGN KEY (`IDgoalie2`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Goalies`
--

DROP TABLE IF EXISTS `Goalies`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Goalies` (
  `playerID` varchar(255) NOT NULL,
  `year` int(11) NOT NULL,
  `stint` int(11) NOT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `GP` varchar(255) DEFAULT NULL,
  `Min` varchar(255) DEFAULT NULL,
  `W` varchar(255) DEFAULT NULL,
  `L` varchar(255) DEFAULT NULL,
  `T/OL` varchar(255) DEFAULT NULL,
  `ENG` varchar(255) DEFAULT NULL,
  `SHO` varchar(255) DEFAULT NULL,
  `GA` varchar(255) DEFAULT NULL,
  `SA` varchar(255) DEFAULT NULL,
  `PostGP` varchar(255) DEFAULT NULL,
  `PostMin` varchar(255) DEFAULT NULL,
  `PostW` varchar(255) DEFAULT NULL,
  `PostL` varchar(255) DEFAULT NULL,
  `PostT` varchar(255) DEFAULT NULL,
  `PostENG` varchar(255) DEFAULT NULL,
  `PostSHO` varchar(255) DEFAULT NULL,
  `PostGA` varchar(255) DEFAULT NULL,
  `PostSA` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`year`,`stint`),
  KEY `Goalies_year_tmID` (`year`,`tmID`) USING BTREE,
  CONSTRAINT `Goalies_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `Goalies_ibfk_2` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `GoaliesSC`
--

DROP TABLE IF EXISTS `GoaliesSC`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `GoaliesSC` (
  `playerID` varchar(255) NOT NULL,
  `year` int(11) NOT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `GP` int(11) DEFAULT NULL,
  `Min` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `T` int(11) DEFAULT NULL,
  `SHO` int(11) DEFAULT NULL,
  `GA` int(11) DEFAULT NULL,
  PRIMARY KEY (`playerID`,`year`),
  KEY `GoaliesSC_year_tmID` (`year`,`tmID`) USING BTREE,
  CONSTRAINT `GoaliesSC_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `GoaliesSC_ibfk_2` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `GoaliesShootout`
--

DROP TABLE IF EXISTS `GoaliesShootout`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `GoaliesShootout` (
  `playerID` varchar(255) DEFAULT NULL,
  `year` int(11) DEFAULT NULL,
  `stint` int(11) DEFAULT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `SA` int(11) DEFAULT NULL,
  `GA` int(11) DEFAULT NULL,
  KEY `GoaliesShootout_playerID` (`playerID`) USING BTREE,
  KEY `GoaliesShootout_year_tmID` (`year`,`tmID`) USING BTREE,
  CONSTRAINT `GoaliesShootout_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `GoaliesShootout_ibfk_2` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `HOF`
--

DROP TABLE IF EXISTS `HOF`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `HOF` (
  `year` int(11) DEFAULT NULL,
  `hofID` varchar(255) NOT NULL,
  `name` varchar(255) DEFAULT NULL,
  `category` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`hofID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Master`
--

DROP TABLE IF EXISTS `Master`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Master` (
  `playerID` varchar(255) DEFAULT NULL,
  `coachID` varchar(255) DEFAULT NULL,
  `hofID` varchar(255) DEFAULT NULL,
  `firstName` varchar(255) DEFAULT NULL,
  `lastName` varchar(255) NOT NULL,
  `nameNote` varchar(255) DEFAULT NULL,
  `nameGiven` varchar(255) DEFAULT NULL,
  `nameNick` varchar(255) DEFAULT NULL,
  `height` varchar(255) DEFAULT NULL,
  `weight` varchar(255) DEFAULT NULL,
  `shootCatch` varchar(255) DEFAULT NULL,
  `legendsID` varchar(255) DEFAULT NULL,
  `ihdbID` varchar(255) DEFAULT NULL,
  `hrefID` varchar(255) DEFAULT NULL,
  `firstNHL` varchar(255) DEFAULT NULL,
  `lastNHL` varchar(255) DEFAULT NULL,
  `firstWHA` varchar(255) DEFAULT NULL,
  `lastWHA` varchar(255) DEFAULT NULL,
  `pos` varchar(255) DEFAULT NULL,
  `birthYear` varchar(255) DEFAULT NULL,
  `birthMon` varchar(255) DEFAULT NULL,
  `birthDay` varchar(255) DEFAULT NULL,
  `birthCountry` varchar(255) DEFAULT NULL,
  `birthState` varchar(255) DEFAULT NULL,
  `birthCity` varchar(255) DEFAULT NULL,
  `deathYear` varchar(255) DEFAULT NULL,
  `deathMon` varchar(255) DEFAULT NULL,
  `deathDay` varchar(255) DEFAULT NULL,
  `deathCountry` varchar(255) DEFAULT NULL,
  `deathState` varchar(255) DEFAULT NULL,
  `deathCity` varchar(255) DEFAULT NULL,
  KEY `Master_coachID` (`coachID`) USING BTREE,
  KEY `Master_hofID` (`hofID`) USING BTREE,
  KEY `Master_playerID` (`playerID`) USING BTREE,
  CONSTRAINT `Master_ibfk_1` FOREIGN KEY (`coachID`) REFERENCES `Coaches` (`coachID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Scoring`
--

DROP TABLE IF EXISTS `Scoring`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Scoring` (
  `playerID` varchar(255) DEFAULT NULL,
  `year` int(11) DEFAULT NULL,
  `stint` int(11) DEFAULT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `pos` varchar(255) DEFAULT NULL,
  `GP` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `A` int(11) DEFAULT NULL,
  `Pts` int(11) DEFAULT NULL,
  `PIM` int(11) DEFAULT NULL,
  `+/-` varchar(255) DEFAULT NULL,
  `PPG` varchar(255) DEFAULT NULL,
  `PPA` varchar(255) DEFAULT NULL,
  `SHG` varchar(255) DEFAULT NULL,
  `SHA` varchar(255) DEFAULT NULL,
  `GWG` varchar(255) DEFAULT NULL,
  `GTG` varchar(255) DEFAULT NULL,
  `SOG` varchar(255) DEFAULT NULL,
  `PostGP` varchar(255) DEFAULT NULL,
  `PostG` varchar(255) DEFAULT NULL,
  `PostA` varchar(255) DEFAULT NULL,
  `PostPts` varchar(255) DEFAULT NULL,
  `PostPIM` varchar(255) DEFAULT NULL,
  `Post+/-` varchar(255) DEFAULT NULL,
  `PostPPG` varchar(255) DEFAULT NULL,
  `PostPPA` varchar(255) DEFAULT NULL,
  `PostSHG` varchar(255) DEFAULT NULL,
  `PostSHA` varchar(255) DEFAULT NULL,
  `PostGWG` varchar(255) DEFAULT NULL,
  `PostSOG` varchar(255) DEFAULT NULL,
  KEY `Scoring_playerID` (`playerID`) USING BTREE,
  KEY `Scoring_year_tmID` (`year`,`tmID`) USING BTREE,
  CONSTRAINT `Scoring_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `Scoring_ibfk_2` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ScoringSC`
--

DROP TABLE IF EXISTS `ScoringSC`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ScoringSC` (
  `playerID` varchar(255) DEFAULT NULL,
  `year` int(11) DEFAULT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `pos` varchar(255) DEFAULT NULL,
  `GP` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `A` int(11) DEFAULT NULL,
  `Pts` int(11) DEFAULT NULL,
  `PIM` int(11) DEFAULT NULL,
  KEY `ScoringSC_playerID` (`playerID`) USING BTREE,
  KEY `ScoringSC_year_tmID` (`year`,`tmID`) USING BTREE,
  CONSTRAINT `ScoringSC_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `ScoringSC_ibfk_2` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ScoringShootout`
--

DROP TABLE IF EXISTS `ScoringShootout`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ScoringShootout` (
  `playerID` varchar(255) DEFAULT NULL,
  `year` int(11) DEFAULT NULL,
  `stint` int(11) DEFAULT NULL,
  `tmID` varchar(255) DEFAULT NULL,
  `S` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `GDG` int(11) DEFAULT NULL,
  KEY `ScoringShootout_playerID` (`playerID`) USING BTREE,
  KEY `ScoringShootout_year_tmID` (`year`,`tmID`) USING BTREE,
  CONSTRAINT `ScoringShootout_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `ScoringShootout_ibfk_2` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `ScoringSup`
--

DROP TABLE IF EXISTS `ScoringSup`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `ScoringSup` (
  `playerID` varchar(255) DEFAULT NULL,
  `year` int(11) DEFAULT NULL,
  `PPA` varchar(255) DEFAULT NULL,
  `SHA` varchar(255) DEFAULT NULL,
  KEY `ScoringSup_playerID` (`playerID`) USING BTREE,
  CONSTRAINT `ScoringSup_ibfk_1` FOREIGN KEY (`playerID`) REFERENCES `Master` (`playerID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `SeriesPost`
--

DROP TABLE IF EXISTS `SeriesPost`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `SeriesPost` (
  `year` int(11) DEFAULT NULL,
  `round` varchar(255) DEFAULT NULL,
  `series` varchar(255) DEFAULT NULL,
  `tmIDWinner` varchar(255) DEFAULT NULL,
  `lgIDWinner` varchar(255) DEFAULT NULL,
  `tmIDLoser` varchar(255) DEFAULT NULL,
  `lgIDLoser` varchar(255) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `T` int(11) DEFAULT NULL,
  `GoalsWinner` int(11) DEFAULT NULL,
  `GoalsLoser` int(11) DEFAULT NULL,
  `note` varchar(255) DEFAULT NULL,
  KEY `SeriesPost_year_tmIDWinner` (`year`,`tmIDWinner`) USING BTREE,
  KEY `SeriesPost_year_tmIDLoser` (`year`,`tmIDLoser`) USING BTREE,
  CONSTRAINT `SeriesPost_ibfk_1` FOREIGN KEY (`year`, `tmIDWinner`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `SeriesPost_ibfk_2` FOREIGN KEY (`year`, `tmIDLoser`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TeamSplits`
--

DROP TABLE IF EXISTS `TeamSplits`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TeamSplits` (
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `tmID` varchar(255) NOT NULL,
  `hW` int(11) DEFAULT NULL,
  `hL` int(11) DEFAULT NULL,
  `hT` int(11) DEFAULT NULL,
  `hOTL` varchar(255) DEFAULT NULL,
  `rW` int(11) DEFAULT NULL,
  `rL` int(11) DEFAULT NULL,
  `rT` int(11) DEFAULT NULL,
  `rOTL` varchar(255) DEFAULT NULL,
  `SepW` varchar(255) DEFAULT NULL,
  `SepL` varchar(255) DEFAULT NULL,
  `SepT` varchar(255) DEFAULT NULL,
  `SepOL` varchar(255) DEFAULT NULL,
  `OctW` varchar(255) DEFAULT NULL,
  `OctL` varchar(255) DEFAULT NULL,
  `OctT` varchar(255) DEFAULT NULL,
  `OctOL` varchar(255) DEFAULT NULL,
  `NovW` varchar(255) DEFAULT NULL,
  `NovL` varchar(255) DEFAULT NULL,
  `NovT` varchar(255) DEFAULT NULL,
  `NovOL` varchar(255) DEFAULT NULL,
  `DecW` varchar(255) DEFAULT NULL,
  `DecL` varchar(255) DEFAULT NULL,
  `DecT` varchar(255) DEFAULT NULL,
  `DecOL` varchar(255) DEFAULT NULL,
  `JanW` int(11) DEFAULT NULL,
  `JanL` int(11) DEFAULT NULL,
  `JanT` int(11) DEFAULT NULL,
  `JanOL` varchar(255) DEFAULT NULL,
  `FebW` int(11) DEFAULT NULL,
  `FebL` int(11) DEFAULT NULL,
  `FebT` int(11) DEFAULT NULL,
  `FebOL` varchar(255) DEFAULT NULL,
  `MarW` varchar(255) DEFAULT NULL,
  `MarL` varchar(255) DEFAULT NULL,
  `MarT` varchar(255) DEFAULT NULL,
  `MarOL` varchar(255) DEFAULT NULL,
  `AprW` varchar(255) DEFAULT NULL,
  `AprL` varchar(255) DEFAULT NULL,
  `AprT` varchar(255) DEFAULT NULL,
  `AprOL` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`year`,`tmID`),
  KEY `TeamSplits_year` (`year`) USING BTREE,
  CONSTRAINT `TeamSplits_ibfk_1` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TeamVsTeam`
--

DROP TABLE IF EXISTS `TeamVsTeam`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TeamVsTeam` (
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `tmID` varchar(255) NOT NULL,
  `oppID` varchar(255) NOT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `T` int(11) DEFAULT NULL,
  `OTL` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`year`,`tmID`,`oppID`),
  KEY `TeamVsTeam_year` (`year`) USING BTREE,
  KEY `TeamVsTeam_oppID_year` (`oppID`,`year`) USING BTREE,
  CONSTRAINT `TeamVsTeam_ibfk_1` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `TeamVsTeam_ibfk_2` FOREIGN KEY (`oppID`, `year`) REFERENCES `Teams` (`tmID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `Teams`
--

DROP TABLE IF EXISTS `Teams`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `Teams` (
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `tmID` varchar(255) NOT NULL,
  `franchID` varchar(255) DEFAULT NULL,
  `confID` varchar(255) DEFAULT NULL,
  `divID` varchar(255) DEFAULT NULL,
  `rank` int(11) DEFAULT NULL,
  `playoff` varchar(255) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `T` int(11) DEFAULT NULL,
  `OTL` varchar(255) DEFAULT NULL,
  `Pts` int(11) DEFAULT NULL,
  `SoW` varchar(255) DEFAULT NULL,
  `SoL` varchar(255) DEFAULT NULL,
  `GF` int(11) DEFAULT NULL,
  `GA` int(11) DEFAULT NULL,
  `name` varchar(255) DEFAULT NULL,
  `PIM` varchar(255) DEFAULT NULL,
  `BenchMinor` varchar(255) DEFAULT NULL,
  `PPG` varchar(255) DEFAULT NULL,
  `PPC` varchar(255) DEFAULT NULL,
  `SHA` varchar(255) DEFAULT NULL,
  `PKG` varchar(255) DEFAULT NULL,
  `PKC` varchar(255) DEFAULT NULL,
  `SHF` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`year`,`tmID`),
  KEY `Teams_tmID` (`tmID`) USING BTREE,
  KEY `Teams_year` (`year`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TeamsHalf`
--

DROP TABLE IF EXISTS `TeamsHalf`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TeamsHalf` (
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `tmID` varchar(255) NOT NULL,
  `half` int(11) NOT NULL,
  `rank` int(11) DEFAULT NULL,
  `G` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `T` int(11) DEFAULT NULL,
  `GF` int(11) DEFAULT NULL,
  `GA` int(11) DEFAULT NULL,
  PRIMARY KEY (`year`,`tmID`,`half`),
  KEY `TeamsHalf_tmID` (`tmID`) USING BTREE,
  KEY `TeamsHalf_year` (`year`) USING BTREE,
  CONSTRAINT `TeamsHalf_ibfk_1` FOREIGN KEY (`tmID`, `year`) REFERENCES `Teams` (`tmID`, `year`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TeamsPost`
--

DROP TABLE IF EXISTS `TeamsPost`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TeamsPost` (
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `tmID` varchar(255) NOT NULL,
  `G` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `T` int(11) DEFAULT NULL,
  `GF` int(11) DEFAULT NULL,
  `GA` int(11) DEFAULT NULL,
  `PIM` varchar(255) DEFAULT NULL,
  `BenchMinor` varchar(255) DEFAULT NULL,
  `PPG` varchar(255) DEFAULT NULL,
  `PPC` varchar(255) DEFAULT NULL,
  `SHA` varchar(255) DEFAULT NULL,
  `PKG` varchar(255) DEFAULT NULL,
  `PKC` varchar(255) DEFAULT NULL,
  `SHF` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`year`,`tmID`),
  KEY `TeamsPost_year` (`year`) USING BTREE,
  CONSTRAINT `TeamsPost_ibfk_1` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `TeamsSC`
--

DROP TABLE IF EXISTS `TeamsSC`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `TeamsSC` (
  `year` int(11) NOT NULL,
  `lgID` varchar(255) DEFAULT NULL,
  `tmID` varchar(255) NOT NULL,
  `G` int(11) DEFAULT NULL,
  `W` int(11) DEFAULT NULL,
  `L` int(11) DEFAULT NULL,
  `T` int(11) DEFAULT NULL,
  `GF` int(11) DEFAULT NULL,
  `GA` int(11) DEFAULT NULL,
  `PIM` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`year`,`tmID`),
  KEY `TeamsSC_year` (`year`) USING BTREE,
  CONSTRAINT `TeamsSC_ibfk_1` FOREIGN KEY (`year`, `tmID`) REFERENCES `Teams` (`year`, `tmID`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `abbrev`
--

DROP TABLE IF EXISTS `abbrev`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `abbrev` (
  `Type` varchar(255) NOT NULL,
  `Code` varchar(255) NOT NULL,
  `Fullname` varchar(255) DEFAULT NULL,
  PRIMARY KEY (`Type`,`Code`)
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

-- Dump completed on 2021-02-22 16:42:43
