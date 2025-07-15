from rest_framework import serializers

class ClassificationRequestSerializer(serializers.Serializer):
    image = serializers.ImageField()

class ClassificationResponseSerializer(serializers.Serializer):
    model = serializers.CharField()
    label = serializers.CharField()
    confidence = serializers.FloatField()
    warning = serializers.BooleanField()
    