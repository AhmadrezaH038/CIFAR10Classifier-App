from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import (
    ClassificationRequestSerializer, 
    ClassificationResponseSerializer
)
from classifier.inferences import classify_image


class ClassifyAPIView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        req_ser = ClassificationRequestSerializer(data=request.data)
        if not req_ser.is_valid():
            return Response(req_ser.errors, status=status.HTTP_400_BAD_REQUEST)
        
        image = req_ser.validated_data['image']
        result = classify_image(request.user, image)

        resp_ser = ClassificationResponseSerializer(data=result)
        if resp_ser.is_valid():
            return Response(resp_ser.data)
        return Response(resp_ser.errors, status.HTTP_500_INTERNAL_SERVER_ERROR)