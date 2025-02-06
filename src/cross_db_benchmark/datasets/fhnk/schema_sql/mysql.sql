-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: FNHK
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
-- Table structure for table `pripady`
--

DROP TABLE IF EXISTS `pripady`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `pripady` (
  `Identifikace_pripadu` int(11) NOT NULL,
  `Identifikator_pacienta` int(11) NOT NULL,
  `Kod_zdravotni_pojistovny` int(11) NOT NULL,
  `Datum_prijeti` date NOT NULL,
  `Datum_propusteni` date NOT NULL,
  `Delka_hospitalizace` int(11) NOT NULL,
  `Vekovy_Interval_Pacienta` varchar(255) NOT NULL,
  `Pohlavi_pacienta` char(1) NOT NULL,
  `Zakladni_diagnoza` varchar(255) NOT NULL,
  `Seznam_vedlejsich_diagnoz` varchar(255) NOT NULL,
  `DRG_skupina` int(11) NOT NULL,
  `PSC` char(5) DEFAULT NULL,
  PRIMARY KEY (`Identifikace_pripadu`),
  KEY `pripady__identifikator_pacienta_indx` (`Identifikator_pacienta`),
  KEY `pripady__datum_prijeti_indx` (`Datum_prijeti`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `vykony`
--

DROP TABLE IF EXISTS `vykony`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `vykony` (
  `Identifikace_pripadu` int(11) NOT NULL,
  `Datum_provedeni_vykonu` date NOT NULL,
  `Typ_polozky` int(11) NOT NULL,
  `Kod_polozky` int(11) NOT NULL,
  `Pocet` int(11) NOT NULL,
  `Body` int(11) NOT NULL,
  PRIMARY KEY (`Identifikace_pripadu`,`Datum_provedeni_vykonu`,`Kod_polozky`),
  KEY `vykony_Identifikace_pripadu` (`Identifikace_pripadu`) USING BTREE,
  CONSTRAINT `vykony_ibfk_1` FOREIGN KEY (`Identifikace_pripadu`) REFERENCES `pripady` (`Identifikace_pripadu`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `zup`
--

DROP TABLE IF EXISTS `zup`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `zup` (
  `Identifikace_pripadu` int(11) NOT NULL,
  `Datum_provedeni_vykonu` date NOT NULL,
  `Typ_polozky` int(11) DEFAULT NULL,
  `Kod_polozky` int(11) NOT NULL,
  `Pocet` decimal(10,2) DEFAULT NULL,
  `Cena` decimal(10,2) DEFAULT NULL,
  PRIMARY KEY (`Identifikace_pripadu`,`Datum_provedeni_vykonu`,`Kod_polozky`),
  KEY `zup_Identifikace_pripadu` (`Identifikace_pripadu`) USING BTREE,
  CONSTRAINT `zup_ibfk_1` FOREIGN KEY (`Identifikace_pripadu`) REFERENCES `pripady` (`Identifikace_pripadu`) ON DELETE CASCADE ON UPDATE CASCADE
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

-- Dump completed on 2021-02-25 14:52:43
