-- MySQL dump 10.13  Distrib 8.0.23, for Linux (x86_64)
--
-- Host: relational.fit.cvut.cz    Database: Accidents
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
-- Table structure for table `nesreca`
--

DROP TABLE IF EXISTS `nesreca`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `nesreca` (
  `id_nesreca` char(6) NOT NULL,
  `klas_nesreca` char(1) NOT NULL,
  `upravna_enota` char(4) NOT NULL,
  `cas_nesreca` datetime NOT NULL,
  `naselje_ali_izven` char(1) NOT NULL,
  `kategorija_cesta` char(1) DEFAULT NULL,
  `oznaka_cesta_ali_naselje` char(5) NOT NULL,
  `tekst_cesta_ali_naselje` varchar(25) NOT NULL,
  `oznaka_odsek_ali_ulica` char(5) NOT NULL,
  `tekst_odsek_ali_ulica` varchar(25) NOT NULL,
  `stacionazna_ali_hisna_st` varchar(9) DEFAULT NULL,
  `opis_prizorisce` char(1) NOT NULL,
  `vzrok_nesreca` char(2) NOT NULL,
  `tip_nesreca` char(2) NOT NULL,
  `vreme_nesreca` char(1) NOT NULL,
  `stanje_promet` char(1) NOT NULL,
  `stanje_vozisce` char(2) NOT NULL,
  `stanje_povrsina_vozisce` char(2) NOT NULL,
  `x` int(11) DEFAULT NULL,
  `y` int(11) DEFAULT NULL,
  `x_wgs84` double DEFAULT NULL,
  `y_wgs84` double DEFAULT NULL,
  PRIMARY KEY (`id_nesreca`),
  KEY `nesreca_cas_nesreca_idx` (`cas_nesreca`),
  KEY `nesreca_pos_idx` (`x`,`y`),
  KEY `nesreca_pos_wgs_84` (`x_wgs84`,`y_wgs84`),
  KEY `upravna_enota` (`upravna_enota`),
  CONSTRAINT `nesreca_ibfk_1` FOREIGN KEY (`upravna_enota`) REFERENCES `upravna_enota` (`id_upravna_enota`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `oseba`
--

DROP TABLE IF EXISTS `oseba`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `oseba` (
  `id_nesreca` char(6) NOT NULL,
  `povzrocitelj_ali_udelezenec` char(1) DEFAULT NULL,
  `starost` tinyint(3) unsigned DEFAULT NULL,
  `spol` char(1) NOT NULL,
  `upravna_enota` char(4) NOT NULL,
  `drzavljanstvo` char(3) NOT NULL,
  `poskodba` char(1) DEFAULT NULL,
  `vrsta_udelezenca` char(2) DEFAULT NULL,
  `varnostni_pas_ali_celada` char(1) DEFAULT NULL,
  `vozniski_staz_LL` tinyint(3) unsigned DEFAULT NULL,
  `vozniski_staz_MM` tinyint(3) unsigned DEFAULT NULL,
  `alkotest` decimal(3,2) DEFAULT NULL,
  `strokovni_pregled` decimal(3,2) DEFAULT NULL,
  `starost_d` char(1) DEFAULT NULL,
  `vozniski_staz_d` char(1) NOT NULL,
  `alkotest_d` char(1) NOT NULL,
  `strokovni_pregled_d` char(1) NOT NULL,
  KEY `oseba_id_nesreca` (`id_nesreca`),
  KEY `upravna_enota` (`upravna_enota`),
  CONSTRAINT `oseba_ibfk_1` FOREIGN KEY (`id_nesreca`) REFERENCES `nesreca` (`id_nesreca`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `oseba_ibfk_2` FOREIGN KEY (`upravna_enota`) REFERENCES `upravna_enota` (`id_upravna_enota`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Table structure for table `upravna_enota`
--

DROP TABLE IF EXISTS `upravna_enota`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!50503 SET character_set_client = utf8mb4 */;
CREATE TABLE `upravna_enota` (
  `id_upravna_enota` char(4) NOT NULL,
  `ime_upravna_enota` varchar(255) NOT NULL,
  `st_prebivalcev` int(10) unsigned DEFAULT NULL,
  `povrsina` smallint(5) unsigned DEFAULT NULL,
  PRIMARY KEY (`id_upravna_enota`)
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

-- Dump completed on 2021-02-22 16:39:19
