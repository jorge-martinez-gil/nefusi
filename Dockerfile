FROM maven:3.9-eclipse-temurin-11 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline -q
COPY src/ src/
RUN mvn package -DskipTests -q

FROM eclipse-temurin:11-jre
WORKDIR /app
COPY --from=builder /app/target/nefusi-1.0.0-jar-with-dependencies.jar nefusi.jar
COPY src/datasets/ datasets/
COPY src/fcl/ fcl/
ENTRYPOINT ["java", "-jar", "nefusi.jar"]
